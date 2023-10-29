import argparse

import jittor as jt
from j_nerf.Module.trainer import Trainer

from j_nerf.Method.config import init_cfg

# jt.flags.gopt_disable=1
jt.flags.use_cuda = 1


def demo():
    config_file = '../j-nerf/j_nerf/Config/ngp_fox.py'
    task = 'train'
    assert task in ['train', 'test', 'render']

    assert (
        jt.flags.cuda_archs[0] >= 61
    ), "Failed: Sm arch version is too low! Sm arch version must not be lower than sm_61!"
    parser = argparse.ArgumentParser(description="Jittor Object Detection Training")
    parser.add_argument(
        "--config-file",
        default=config_file,
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--task",
        default=task,
        help="train,val,test",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="",
        type=str,
    )

    parser.add_argument(
        "--type",
        default="novel_view",
        type=str,
    )
    parser.add_argument(
        "--mcube_threshold",
        default=0.0,
        type=float,
    )

    args = parser.parse_args()

    assert args.type in [
        "novel_view",
        "mesh",
    ], f"{args.type} not support, please choose [novel_view, mesh]"
    assert args.task in [
        "train",
        "test",
        "render",
        "validate_mesh",
    ], f"{args.task} not support, please choose [train, test, render, validate_mesh]"
    is_continue = False
    if args.task == "validate_mesh":
        is_continue = True

    if args.config_file:
        init_cfg(args.config_file)

    if args.type == "novel_view":
        trainer = Trainer()

    if args.task == "train":
        trainer.train()
    elif args.task == "test":
        trainer.test(True)
    elif args.task == "render":
        trainer.render(True, args.save_dir)
    elif args.task == "validate_mesh":
        trainer.validate_mesh(
            world_space=False, resolution=512, threshold=args.mcube_threshold
        )
    return True
