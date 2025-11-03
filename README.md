# IONA_Pose_Estimation

## Setup PRocesdure

```
conda create -n PoseEstimation python=3.9
conda activate PoseEstimation
conda config --set channel_priority flexible
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
git clone https://github.com/NVlabs/FoundationPose.git
python -m pip install -r FoundationPose/requirements.txt
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```
