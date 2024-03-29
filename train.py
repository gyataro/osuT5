from accelerate import Accelerator
from accelerate.utils import LoggerType, ProjectConfiguration
from omegaconf import open_dict, DictConfig
import hydra
import torch
import time

from osuT5.utils import (
    setup_args,
    train,
    get_config,
    get_model,
    get_tokenizer,
    get_scheduler,
    get_optimizer,
    get_dataloaders,
)


@hydra.main(config_path="configs", config_name="train", version_base="1.1")
def main(args: DictConfig):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        log_with=LoggerType.TENSORBOARD,
        project_config=ProjectConfiguration(
            project_dir=".", logging_dir="tensorboard_logs"
        ),
    )
    accelerator.init_trackers("project")

    setup_args(args)

    config = get_config(args)
    model = get_model(config)
    tokenizer = get_tokenizer()
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args)

    (
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    if args.checkpoint_path:
        accelerator.load_state(args.checkpoint_path)

    if args.compile:
        model = torch.compile(model)

    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()

    train(
        model,
        train_dataloader,
        test_dataloader,
        accelerator,
        scheduler,
        optimizer,
        tokenizer,
        args,
    )


if __name__ == "__main__":
    main()
