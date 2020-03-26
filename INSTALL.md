## Installation

### Requirements

- Linux (tested on Ubuntu 16.04 and CentOS 7.2)
- Python 3.4+
- PyTorch 1.0
- Cython
- [mmcv](https://github.com/open-mmlab/mmcv) >= 0.2.2

### Install mmdetection

a. Install PyTorch 1.0 and torchvision following the [official instructions](https://pytorch.org/).

b. Clone the mmdetection repository.

```shell
git clone https://github.com/open-mmlab/mmdetection.git
```

c. Compile cuda extensions.

```shell
cd mmdetection
pip install cython  # or "conda install cython" if you prefer conda
./compile.sh  # or "PYTHON=python3 ./compile.sh" if you use system python3 without virtual environments
```

d. Install mmdetection (other dependencies will be installed automatically).

```shell
python(3) setup.py install  # add --user if you want to install it locally
# or "pip install ."
```



### Intsall with Conda

```shell
conda create -n open-mmlab python=3.7 -y
source activate open-mmlab

conda install -c pytorch pytorch torchvision -y
conda install cython -y
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
```
see more details at [mmdetection](https://github.com/open-mmlab/mmdetection)

