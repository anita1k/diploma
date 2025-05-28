import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout
import os
import hydra
from omegaconf import OmegaConf
import trainers
import wandb


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))

def worker_main(rank, world_size, config, policy, reference_model):
    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, 'BasicTrainer')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    OmegaConf.resolve(config)

    missing_keys = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.eval_every % config.batch_size != 0:
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    model_kwargs = {'device_map': 'balanced'}
    policy_dtype = getattr(torch, 'float32')
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        "./facebook", cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    reference_model_dtype = getattr(torch, 'float16')
    reference_model = transformers.AutoModelForCausalLM.from_pretrained(
        "./facebook", cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
    disable_dropout(reference_model)
    worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()