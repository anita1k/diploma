import torch
import torch.nn.functional as F
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from prepare_data import get_batch_iterator
from utils import slice_and_move_batch_for_device, pad_to_length, get_local_dir
import numpy as np
import wandb
import tqdm
import random
import os
from collections import defaultdict
import time
import json


def preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta, label_smoothing = 0.0, ipo = False):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2
    else:
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits, labels, average_log_prob = False):
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch):
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy, config, seed, run_dir, reference_model = None, rank = 0, world_size = 1):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
        )

        self.policy = policy
        self.reference_model = reference_model
        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size)
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size)
        self.eval_batches = list(self.eval_iterator)

    def get_batch_samples(self, batch):
        policy_output = self.policy.generate(
            batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        reference_output = self.reference_model.generate(
            batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)
        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        return policy_output_decoded, reference_output_decoded
    
    def concatenated_forward(self, model, batch):
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch, loss_config, train=True):
        metrics = {}
        train_test = 'train' if train else 'eval'

        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)
            
        if loss_config.name == 'dpo':
            loss_kwargs = {'beta': loss_config.beta, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
        elif loss_config.name == 'ipo':
            loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
        else:
            raise ValueError(f'unknown loss {loss_config.name}')

        losses, chosen_rewards, rejected_rewards = preference_loss(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
        metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = losses.detach()
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
        return losses.mean(), metrics
    

    def train(self):
        torch.cuda.empty_cache()
        
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            # evaluation
            if self.example_counter % self.config.eval_every == 0:
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                            reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)
                    
                    data = {
                            "step": self.example_counter,
                            **mean_eval_metrics
                        }
                        
                    self.load_json(data, file_path=f'metrics_eval_{self.config.loss.beta}.json')

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

            # training
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter

                if self.config.wandb.enabled and self.rank == 0:
                    if self.example_counter % self.config.n_eval_examples == 0:
                        wandb.log(mean_train_metrics, step=self.example_counter)
                        
                        data = {
                            "step": self.example_counter,
                            **mean_train_metrics
                        }
                        
                        self.load_json(data, file_path=f'metrics_train_{self.config.loss.beta}.json')

                last_log = time.time()


    def clip_gradient(self):
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()
    
    def load_json(self, data, file_path):
        try:
            if isinstance(file_path, list):
                file_path = file_path[0]
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            existing_data = [] 
        if isinstance(existing_data, list):
            existing_data.append(data)
        else:
            print("Error: data is not the list.")
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4, separators=(",", ":"))
    
    def write_state_dict(self, step, state, metrics, filename, dir_name):
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir, metrics = None):
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
  