import jittor as jt
from j_nerf.Method.config import get_cfg
from j_nerf.Model.grid_encode import GridEncode
from j_nerf.Method.global_vars import proj_options
from jittor import nn
from j_nerf.Method.error_check import checkError


class HashEncoder(nn.Module):
    def __init__(
        self,
        n_pos_dims=3,
        n_features_per_level=2,
        n_levels=32,
        base_resolution=16,
        log2_hashmap_size=19,
    ):
        self.cfg = get_cfg()
        aabb_scale = self.cfg.dataset_obj.aabb_scale
        self.hash_func = self.cfg.hash_func
        self.hash_func_header = f"""
#define get_index(p0,p1,p2) {self.hash_func}
        """
        self.encoder = GridEncode(
            self.hash_func_header,
            aabb_scale=aabb_scale,
            n_pos_dims=n_pos_dims,
            n_features_per_level=n_features_per_level,
            n_levels=n_levels,
            base_resolution=base_resolution,
            log2_hashmap_size=log2_hashmap_size,
        )
        self.grad_type = "float16"
        self.m_grid = jt.init.uniform(
            [self.encoder.m_n_params], low=-1e-4, high=1e-4, dtype=self.grad_type
        )
        self.out_dim = n_features_per_level * n_levels
        header_path = "../j-nerf/j_nerf/Cpp/hash_encoder/"
        proj_options[f"FLAGS: -I{header_path}"] = 1
        return

    def execute(self, x):
        checkError(x, "HashEncoder.execute.x")
        assert self.m_grid.dtype == self.grad_type, (
            self.m_grid.dtype,
            "!=",
            self.grad_type,
        )
        output = self.encoder(x, self.m_grid)
        # print("hash.output.shape:", output.shape)
        checkError(output, "HashEncoder.execute.output")
        assert output.dtype == self.grad_type
        return output
