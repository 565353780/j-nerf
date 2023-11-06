exp_name = "Scar"
dataset_type = "NerfDataset"
dataset_dir = "my/data/" + exp_name
dataset_aabb = {"Car": 4, "Coffee": 1, "Easyship": 8, "Scar": 5, "Scarf": 8}
dataset_scale = {
    "Car": None,
    "Coffee": None,
    "Easyship": None,
    "Scar": None,
    "Scarf": 0.05,
}
dataset_offset = {
    "Car": [-2.0, -0.5, 0.0],
    "Coffee": None,
    "Easyship": None,
    "Scar": None,
    "Scarf": None,
}

dataset = dict(
    train=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1, -1, 1],
        batch_size=4096,
        mode="train",
        aabb_scale=dataset_aabb[exp_name],
        scale=dataset_scale[exp_name],
        offset=dataset_offset[exp_name],
    ),
    val=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1, -1, 1],
        batch_size=4096,
        mode="val",
        preload_shuffle=False,
        aabb_scale=dataset_aabb[exp_name],
        scale=dataset_scale[exp_name],
        offset=dataset_offset[exp_name],
    ),
    test=dict(
        type=dataset_type,
        root_dir=dataset_dir,
        correct_pose=[-1, -1, 1],
        batch_size=4096,
        mode="test",
        have_img=False,
        H=800,
        W=800,
        preload_shuffle=False,
        aabb_scale=dataset_aabb[exp_name],
        scale=dataset_scale[exp_name],
        offset=dataset_offset[exp_name],
    ),
)

background_color = [1, 1, 1]
const_dt = True
