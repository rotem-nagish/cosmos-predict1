# nvrtc_check.py
import os
import shutil

os.environ["PATH"] += os.pathsep + "/home/rotem/miniforge3/envs/cosmos-predict1/bin"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

# enable PyTorch NVRTC debug output (try both names)
os.environ["TORCH_NVRTC_DEBUG"] = "1"
os.environ["PYTORCH_NVRTC_DEBUG"] = "1"

from torch.utils.cpp_extension import load_inline
cuda_src = r'''
extern "C" __global__ void hello_kernel(float *x) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  x[idx] = 0.0f;
}
'''
try:
    print("Starting inline compile (verbose)...")
    load_inline(
        name="nvrtc_test",
        cpp_sources="",
        cuda_sources=cuda_src,
        functions=["hello_kernel"],
        verbose=True,
    )
    print("Compile succeeded.")
except Exception as e:
    print("Compile failed — exception follows:")
    import traceback
    traceback.print_exc()
