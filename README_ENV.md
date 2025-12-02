# Environment Configuration

This repository uses environment variables for configuration to make it portable across different systems.

## Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your system-specific paths:**
   ```bash
   # Edit the file with your preferred editor
   nano .env  # or vim, code, etc.
   ```

3. **(Optional) Install python-dotenv for automatic .env loading:**
   ```bash
   pip install python-dotenv
   ```

   If `python-dotenv` is not installed, you can also set environment variables manually before running Python:
   ```bash
   export NVTE_CUDA_INCLUDE_DIR=/usr/local/cuda-12.8/include
   # ... other variables
   python your_script.py
   ```

## Configuration Variables

### CUDA Configuration
- `NVTE_CUDA_INCLUDE_DIR`: Path to CUDA include directory
- `LD_LIBRARY_PATH`: Paths to CUDA and system libraries
- `TORCH_CUDA_ARCH_LIST`: CUDA architecture version
- `TORCH_NVCC_FLAGS`: NVCC compiler flags

### Python Environment
- `PYTHON_ENV_BIN`: Path to your conda/venv bin directory (optional)

### Distributed Training
- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`: Multi-GPU training configuration
- `MASTER_ADDR`, `MASTER_PORT`: Distributed training network settings
- `NCCL_SHM_DISABLE`: NCCL shared memory settings

### Other
- `TORCH_HOME`: Torch model cache directory (optional)

## How It Works

The `cosmos_predict1/__init__.py` module:
1. Attempts to load `.env` if `python-dotenv` is installed
2. Reads environment variables with `os.getenv()`
3. Only sets values if they're defined in the environment
4. Uses sensible defaults for distributed training variables

This approach allows:
- Each developer to have their own `.env` file (git-ignored)
- Override from shell environment when needed
- Training scripts to override these settings
- Clean separation of code and configuration