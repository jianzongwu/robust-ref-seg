# Installation

We recommend using Conda to manage the environment.

## 1. Create a new python environment.

We tested our code with python version=3.10. Other versions **may** be compatible.

``` bash
conda create -n rris python=3.10
conda activate rris
```

## 2. Install pytorch

Our repo is built upon pytorch=1.12.1, torchvision=0.13.1, torchaudio=0.12.1, and cudatoolkit=11.6.0.  Other versions **may** be compatible.

``` bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

## 3. Install mmcv

We recommend install mmcv locally. A local installation will have lower probabilities to trigger errors.
Library repositories like mmcv will be stored in the `libs` folder.
In this repo, we use mmcv-full=1.7.1 and install it locally. Other installation methods can be found in [mmcv official repo](https://github.com/open-mmlab/mmcv).

``` bash
mkdir libs
cd libs
git clone git@github.com:open-mmlab/mmcv.git
cd mmcv
pip install -r requirements/optional.txt
pip install -e . -v
```

Then evaluate the installation.

``` bash
python .dev_scripts/check_installation.py
```

## 4. Install other dependencies

Install pycocotools locally from [this repo](https://github.com/cocodataset/cocoapi.git).

``` bash
cd ..
git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI
pip install -e .
``` 

Then install other dependencies from `requirements.txt`.

``` bash
pip install -r requirements.txt
```