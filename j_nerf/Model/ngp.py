import jittor as jt
from j_nerf.Config.registry import ENCODERS, NETWORKS
from j_nerf.Method.config import get_cfg
from j_nerf.Method.registry import build_from_cfg
from jittor import nn

from j_nerf.Model.fmlp import FMLP


@NETWORKS.register_module()
class NGPNetworks(nn.Module):
    def __init__(
        self,
        use_fully=True,
        density_hidden_layer=1,
        density_n_neurons=64,
        rgb_hidden_layer=2,
        rgb_n_neurons=64,
    ):
        super(NGPNetworks, self).__init__()
        self.use_fully = use_fully
        self.cfg = get_cfg()
        self.using_fp16 = self.cfg.fp16
        self.pos_encoder = build_from_cfg(self.cfg.encoder.pos_encoder, ENCODERS)
        self.dir_encoder = build_from_cfg(self.cfg.encoder.dir_encoder, ENCODERS)

        if self.use_fully and jt.flags.cuda_archs[0] >= 75 and self.using_fp16:
            assert self.pos_encoder.out_dim % 16 == 0
            assert self.dir_encoder.out_dim % 16 == 0
            self.density_mlp = FMLP([self.pos_encoder.out_dim, density_n_neurons, 16])
            self.rgb_mlp = FMLP(
                [self.dir_encoder.out_dim + 16, rgb_n_neurons, rgb_n_neurons, 3]
            )
        else:
            if self.use_fully and not (jt.flags.cuda_archs[0] >= 75):
                print(
                    "Warning: Sm arch is lower than sm_75, FFMLPs is not supported. Automatically use original MLPs instead."
                )
            elif self.use_fully and not self.using_fp16:
                print(
                    "Warning: FFMLPs only support float16. Automatically use original MLPs instead."
                )
            self.density_mlp = nn.Sequential(
                nn.Linear(self.pos_encoder.out_dim, density_n_neurons, bias=False),
                nn.ReLU(),
                nn.Linear(density_n_neurons, 16, bias=False),
            )
            self.rgb_mlp = nn.Sequential(
                nn.Linear(self.dir_encoder.out_dim + 16, rgb_n_neurons, bias=False),
                nn.ReLU(),
                nn.Linear(rgb_n_neurons, rgb_n_neurons, bias=False),
                nn.ReLU(),
                nn.Linear(rgb_n_neurons, 3, bias=False),
            )
        self.set_fp16()

    def execute(self, pos_input, dir_input):
        if self.using_fp16:
            with jt.flag_scope(auto_mixed_precision_level=5):
                return self.execute_(pos_input, dir_input)
        else:
            return self.execute_(pos_input, dir_input)

    def execute_(self, pos_input, dir_input):
        dir_input = self.dir_encoder(dir_input)
        pos_input = self.pos_encoder(pos_input)
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
        if self.using_fp16:
            self.density_mlp.float16()
            self.rgb_mlp.float16()
            self.pos_encoder.float16()
            self.dir_encoder.float16()
