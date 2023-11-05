import jittor as jt
from j_nerf.Method.config import get_cfg
from jittor import nn

from j_nerf.Model.fmlp import FMLP
from j_nerf.Model.hash_encoder import HashEncoder
from j_nerf.Model.sh_encoder import SHEncoder
from j_nerf.Method.error_check import checkError


class NGPNetworks(nn.Module):
    def __init__(
        self,
        density_n_neurons=64,
        rgb_n_neurons=64,
    ):
        super(NGPNetworks, self).__init__()
        self.cfg = get_cfg()

        n_features_per_level = 2
        feature_dim = 16

        self.pos_encoder = HashEncoder(
            n_features_per_level=n_features_per_level, n_levels=feature_dim
        )
        self.dir_encoder = SHEncoder()

        assert self.pos_encoder.out_dim % 16 == 0
        assert self.dir_encoder.out_dim % 16 == 0
        self.density_mlp = FMLP(
            [
                self.pos_encoder.out_dim,
                density_n_neurons,
                self.dir_encoder.out_dim,
            ]
        )
        self.rgb_mlp = FMLP(
            [
                self.dir_encoder.out_dim + self.density_mlp.output_shape1,
                rgb_n_neurons,
                rgb_n_neurons,
                3,
            ]
        )
        self.set_fp16()
        return

    def execute(self, pos_input, dir_input):
        with jt.flag_scope(auto_mixed_precision_level=5):
            return self.execute_(pos_input, dir_input)

    def execute_(self, pos_input, dir_input):
        checkError(dir_input, "NGP.execute_.dir_input")
        checkError(pos_input, "NGP.execute_.pos_input")

        dir_input = self.dir_encoder(dir_input)
        checkError(dir_input, "NGP.execute_.dir_input encoded")

        pos_input = self.pos_encoder(pos_input)
        checkError(pos_input, "NGP.execute_.pos_input encoded")

        density = self.density_mlp(pos_input)
        rgb = jt.concat([density, dir_input], -1)
        rgb = self.rgb_mlp(rgb)
        outputs = jt.concat([rgb, density[..., :1]], -1)  # batchsize 4: rgbd
        return outputs

    def density(self, pos_input):  # batchsize,3
        density = self.pos_encoder(pos_input)
        density = self.density_mlp(density)[:, :1]
        return density

    def set_fp16(self):
        self.density_mlp.float16()
        self.rgb_mlp.float16()
        self.pos_encoder.float16()
        self.dir_encoder.float16()
        return True
