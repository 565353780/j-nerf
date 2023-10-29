import jittor as jt
from j_nerf.Model.fully_fused_mlp import FullyFusedMlp_weight
from jittor import init, nn


class FMLP(nn.Module):
    def __init__(self, weight_shapes, weights=None):
        super(FMLP, self).__init__()
        if weights == None:
            assert len(weight_shapes) > 2
            self.output_shape1 = weight_shapes[-1]
            dweights = []
            for i in range(len(weight_shapes) - 1):
                dweights.append(
                    init.invariant_uniform(
                        (weight_shapes[i], weight_shapes[i + 1]), "float16"
                    ).float16()
                )
        else:
            assert len(weights) >= 2
            self.output_shape1 = weights[-1].shape[-1]
            dweights = weights
        self.func = FullyFusedMlp_weight(dweights)
        con_weights = []
        for i in range(len(dweights)):
            if i == len(dweights) - 1:
                if dweights[i].shape[1] < 16:
                    dweights[i] = jt.concat(
                        [
                            dweights[i],
                            jt.zeros((dweights[i].shape[0], 16 - dweights[i].shape[1])),
                        ],
                        -1,
                    ).float16()
            con_weights.append(dweights[i].transpose(1, 0).reshape(-1))
        jt_con_weights = jt.concat(con_weights, -1)
        self.con_weights = jt_con_weights

    def execute(self, x):
        if x.shape[0] == 0:
            return jt.empty([0, self.output_shape1]).float16()
        ret = self.func(x, self.con_weights)
        if self.output_shape1 != ret.shape[1]:
            ret = ret[:, : self.output_shape1]
        return ret
