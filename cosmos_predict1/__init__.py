# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

# Load .env file if it exists
def _load_dotenv():
    """Load environment variables from .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_file = Path(__file__).parent.parent / '.env'
        if env_file.exists():
            load_dotenv(env_file)
    except ImportError:
        pass  # python-dotenv not installed, will use system env vars

_load_dotenv()

# CUDA Configuration - only set if specified in environment
if cuda_include := os.getenv('NVTE_CUDA_INCLUDE_DIR'):
    os.environ['NVTE_CUDA_INCLUDE_DIR'] = cuda_include

if ld_library_path := os.getenv('LD_LIBRARY_PATH'):
    os.environ['LD_LIBRARY_PATH'] = ld_library_path

if cuda_arch := os.getenv('TORCH_CUDA_ARCH_LIST'):
    os.environ['TORCH_CUDA_ARCH_LIST'] = cuda_arch

if nvcc_flags := os.getenv('TORCH_NVCC_FLAGS'):
    os.environ['TORCH_NVCC_FLAGS'] = nvcc_flags

# Python environment path - append to PATH if specified
if python_env_bin := os.getenv('PYTHON_ENV_BIN'):
    os.environ['PATH'] += os.pathsep + python_env_bin

# Torch cache directory
if torch_home := os.getenv('TORCH_HOME'):
    os.environ['TORCH_HOME'] = torch_home

# Distributed Training Configuration - only set defaults if not already set
# This allows the environment or training scripts to override these
if 'RANK' not in os.environ:
    os.environ['RANK'] = os.getenv('RANK', '0')
if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = os.getenv('LOCAL_RANK', '0')
if 'WORLD_SIZE' not in os.environ:
    os.environ['WORLD_SIZE'] = os.getenv('WORLD_SIZE', '1')
if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', '127.0.0.1')
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = os.getenv('MASTER_PORT', '29500')
if 'NCCL_SHM_DISABLE' not in os.environ:
    os.environ['NCCL_SHM_DISABLE'] = os.getenv('NCCL_SHM_DISABLE', '1')