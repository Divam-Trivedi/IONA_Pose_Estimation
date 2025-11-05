# IONA_Pose_Estimation

## Setup Procesdure

```
conda create -n PoseEstimation python=3.9
conda activate PoseEstimation
conda config --set channel_priority flexible
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
git clone https://github.com/NVlabs/FoundationPose.git
cd FoundationPose
python -m pip install -r requirements.txt
python -m pip install --quiet --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
conda install -c nvidia/label/cuda-11.8.0 cuda-toolkit
pip uninstall setuptools && pip install setuptools==69.5.1
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh

## DETECTRON2
cd ..
python -m pip install pyyaml==5.1
git clone 'https://github.com/facebookresearch/detectron2'
pip install --no-build-isolation -e ./detectron2

## CAMERA
pip install pyrealsense2
```

After doing the CMAKE_PREFIX... command, an updated mycpp folder would be created inside the foudnatonpose directory. COpy paste the folder to root. 

Dowload all network weights for Foundationpose from here [https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing] and put them inside a FP_weights folder.
Download the detectron2 weights from here [link given] and add them in the detectron2_models folder. Update the models infomation text file accoridngly. 


Once everythg is setup adn the folder looks like this:

├── datareader.py
├── detectron2
│   ├── build
│   ├── configs
│   ├── datasets
│   ├── demo
│   ├── detectron2
│   ├── detectron2.egg-info
│   ├── dev
│   ├── docker
│   ├── docs
│   ├── GETTING_STARTED.md
│   ├── INSTALL.md
│   ├── LICENSE
│   ├── MODEL_ZOO.md
│   ├── projects
│   ├── README.md
│   ├── setup.cfg
│   ├── setup.py
│   ├── tests
│   └── tools
├── detectron2_models
│   ├── medical_objects_fruits.pth
│   └── model_info.txt
├── estimater.py
├── FoundationPose
│   ├── assets
│   ├── build_all_conda.sh
│   ├── build_all.sh
│   ├── bundlesdf
│   ├── cuda_11.8.0_520.61.05_linux.run
│   ├── datareader.py
│   ├── debug
│   ├── docker
│   ├── estimater.py
│   ├── learning
│   ├── LICENSE
│   ├── mycpp
│   ├── offscreen_renderer.py
│   ├── __pycache__
│   ├── readme.md
│   ├── requirements.txt
│   ├── run_demo.py
│   ├── run_linemod.py
│   ├── run_ycb_video.py
│   ├── Utils.py
│   └── weights
├── FP_Utils.py
├── FP_weights
│   ├── 2023-10-28-18-33-37
│   └── 2024-01-11-20-02-45
├── learning
│   ├── datasets
│   ├── models
│   └── training
├── Meal_Tray_Scenario
│   └── output_20251105_163145
├── Meshes
│   └── Sb_Cup
├── mycpp
│   ├── build
│   ├── CMakeLists.txt
│   ├── include
│   └── src
├── __pycache__
│   ├── datareader.cpython-39.pyc
│   ├── estimater.cpython-39.pyc
│   ├── FP_Utils.cpython-39.pyc
│   └── Utils.cpython-39.pyc
├── run_real_time_short.py
└── Utils.py


Connect the camera and Run the command
```
python3 run_main.py
```
