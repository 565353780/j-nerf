from j_nerf.Method.config import init_cfg
from j_nerf.Module.trainer import Trainer

def demo():
    config_file = '../j-nerf/j_nerf/Config/ngp_fox.py'
    task = 'train'
    save_dir = '../j-nerf/output/demo.mp4'
    mcube_threshold = 0.0

    assert task in ['train', 'test', 'render', 'validate_mesh']

    init_cfg(config_file)

    trainer = Trainer()

    if task == "train":
        trainer.train()
    elif task == "test":
        trainer.test(True)
    elif task == "render":
        trainer.render(True, save_dir)
    elif task == "validate_mesh":
        trainer.validate_mesh(
            world_space=False, resolution=512, threshold=mcube_threshold
        )
    return True
