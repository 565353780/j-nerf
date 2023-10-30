encoder = dict(
    pos_encoder=dict(
        type="HashEncoder",
    ),
    dir_encoder=dict(
        type="SHEncoder",
    ),
)

exp_name = "fox"
log_dir = "./logs"
tot_train_steps = 40000
# Background color, value range from 0 to 1
background_color = [0, 0, 0]
# Hash encoding function used in Instant-NGP
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
# Expected number of sampling points per batch
target_batch_size = 1 << 18
# Set const_dt=True for higher performance
# Set const_dt=False for faster convergence
const_dt = False
# Use fp16 for faster training
fp16 = True
