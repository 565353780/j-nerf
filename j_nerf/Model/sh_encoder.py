import jittor as jt
from j_nerf.Method.config import get_cfg
from j_nerf.Method.global_vars import proj_options
from jittor import Function


class SHEncoder(Function):
    def __init__(self):
        self.cfg = get_cfg()
        self.num_elements = 4194304
        self.m_n_padded_output_dims = 16
        self.m_sh_degree = 4
        self.m_n_to_pad = 0
        self.grad_type = "float16"
        header_path = "../j-nerf/j_nerf/Cpp/sh_encoder/"
        proj_options[f"FLAGS: -I{header_path}"] = 1
        self.out_dim = self.m_n_padded_output_dims

    def execute(self, x):
        self.num_elements = x.shape[0]

        output = jt.code(
            (self.num_elements, 16),
            self.grad_type,
            [x],
            cuda_header='#include "SphericalEncode.h"',
            cuda_src=f"""
  
       #define grad_t out_type

        uint32_t num_elements=in0_shape0;
        uint32_t m_n_padded_output_dims={self.m_n_padded_output_dims};
        uint32_t m_sh_degree={self.m_sh_degree};
        uint32_t m_n_to_pad={self.m_n_to_pad};
       
        cudaStream_t stream=0;
    
        PitchedPtr<const float> inputs={{in0_p,in0_shape1}};
		PitchedPtr<grad_t> outputs={{out_p,out_shape1}};
		float* dy_dx = nullptr;
        linear_kernel(kernel_sh<grad_t>, 0, stream,
			num_elements,
			m_sh_degree,
			m_n_to_pad,
			inputs,
            outputs,
			dy_dx
		);
        """,
        )
        output.compile_options = proj_options
        return output

    def grad(self, grad_x):
        return None
