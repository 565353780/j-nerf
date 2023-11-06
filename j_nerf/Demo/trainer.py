from j_nerf.Module.trainer import Trainer


def demo():
    exp_name = "NeRF_wine"
    dataset_folder_path = "../colmap-manage/output/" + exp_name + "/jn/"

    task = "train"
    save_dir = "../j-nerf/output/demo.mp4"
    mcube_threshold = 0.0

    assert task in ["train", "test", "render", "validate_mesh"]

    trainer = Trainer(exp_name, dataset_folder_path)

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
