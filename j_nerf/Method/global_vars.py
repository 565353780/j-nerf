import jittor as jt

jt.flags.use_cuda = 1

global_headers = """
#include "pcg32.h"
namespace jittor {
extern int global_var1;
extern pcg32 rng;
}
"""

global_src = """
namespace jittor {
int global_var1 = 123;
pcg32 rng{1337};
}
"""

proj_path = "../j-nerf/j_nerf/"
flags_str = (
    "FLAGS:"
    + " -I"
    + proj_path
    + "Lib/eigen"
    + " -I"
    + proj_path
    + "Lib/include"
    + " -I"
    + proj_path
    + "Lib/pcg32"
    + " -I"
    + proj_path
    + "Cpp/"
    + " -DGLOBAL_VAR"
    + " --extended-lambda"
    + " --expt-relaxed-constexpr"
)
proj_options = {flags_str: 1}
gv = jt.code(
    [1],
    int,
    cuda_header=global_headers + global_src,
    cuda_src="""
""",
)
gv.compile_options = proj_options
gv.sync()
