# Installation

## Step 1: Set Up Conda Environment

```bash
conda create -n seq_multi_grasp python=3.9
conda activate seq_multi_grasp
```

## Step 2: Check CUDA Version

Check your system's installed CUDA version:

```bash
which nvcc
nvcc --version
```

The setup has been tested with CUDA 11.8, CUDA 12.4 and CUDA 12.8. In theory, other CUDA versions should also work as long as PyTorch is compiled against the same version found via `nvcc`.


## Step 3: Install PyTorch (Ensure Compatibility with Your CUDA Version)

```bash
# For CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.4
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda install -c conda-forge libstdcxx-ng
```

## Step 4: Build and Install PyTorch3D

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Step 5: Install ManiSkill3

```bash
pip install git+https://github.com/haosulab/ManiSkill.git@v3.0.0b21
```

## Step 6: Install Additional Dependencies

```bash
pip install -r requirements.txt
```

## Step 7: Install External Dependencies

Some dependencies should be installed separately:

```bash

git submodule update --init --recursive


# Install curobo
cd third-party/curobo
pip install -e . --no-build-isolation

cd ../..

# Install kaolin
cd third-party/kaolin
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
export IGNORE_TORCH_VER=1
pip install -e .

cd ../..


cd third-party/pointnet2_ops_lib
python setup.py develop

cd ../..

cd third-party/allegro_visualization
pip install -e .

cd ../..

```