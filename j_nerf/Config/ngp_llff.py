dataset_type = "LLFFDataset"
dataset_dir = "data/fern"
dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode="train",
        factor=8,
        llffhold=8,
        aabb_scale=64,
    ),
    val=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode="val",
        preload_shuffle=False,
        factor=8,
        llffhold=8,
        aabb_scale=64,
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        batch_size=4096,
        mode="test",
        preload_shuffle=False,
        factor=8,
        llffhold=8,
        aabb_scale=64,
    ),
)

# Load pre-trained model
load_ckpt = False
# path of checkpoint file, None for default path
ckpt_path = None
# test output image with alpha
alpha_image = False
