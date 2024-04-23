# BLDAFT

## Introduction

BLDAFT is a new IR small target detection algorithm named Bilateral Local Discrete Attention Fusion Transformer for infrared small targets detection under maritime.
This open-source code is used for test and verify our paper, or for future comparison.
![BLDAFT](./figs/BLDAFT.png)


## License

This project is released under the [Apache 2.0 license](./LICENSE).

## Installation
Refer to Deformable detr [here](https://github.com/fundamentalvision/Deformable-DETR)

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

### Compiling CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

## Usage

### Dataset preparation

About the dataset's structure please refer to [COCO 2017 dataset](https://cocodataset.org/) and organize them as following:
(Replace coco name with your own dataset.)
```
code_root/
└── data/
    └── coco/
        ├── train2017/
        ├── val2017/
        └── annotations/
        	├── instances_train2017.json
        	└── instances_val2017.json
```

### Training

Refer to Deformable detr [here](https://github.com/fundamentalvision/Deformable-DETR)

### Test

We have provided some test images and run ./util/BLDAFT_Visualize.py with the settings below to test the trained model.
```
--eval
--coco_path
../MTD_test
--resume
../experiments/test.pth
```