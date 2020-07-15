# ReSprop

This repository enables the reproduction of the experiments described in the CVPR 2020 paper:

ReSprop: Reuse Sparsified Backpropagation

Negar Goli, Tor M. Aamodt

[Link to paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Goli_ReSprop_Reuse_Sparsified_Backpropagation_CVPR_2020_paper.pdf)

![ReSprop Demo](demo.gif)


## Requirements 

* Python 3.7.5

* torch 1.1.0

* torchvision

* cuda 10

* gcc 7.5.0

### Prerequisites

The code uses the custom C++ and cuda extensions of pytorch. 
[CUSTOM C++ Pytorch](https://pytorch.org/tutorials/advanced/cpp_extension.html) 

The c++ code is availabe in backward folder. 

Build the costum c++ backward functions in the backward folder: 

```
python setup.py install 
```

Download ImageNet dataset and use uncropped data 

## Running the Algorithm 
The algorithm can be run on up to 8 GPUs

-a : Architecture 
--iterations: Number of iterations 
--sparsity: targeted reuse_sparsity
--warmup: 1 means with warmup phase, 0 without


Example: ImageNet on resnet18 with 80% reuse sparsity and with warmup (100-80=20%). 




```
python main_ImageNet.py -a resnet18 --sparsity 20  --warmup 1 PATH_TO_THE_IMAGENET_DATASET
```

## Code explanation

ReSprop_conv.py is a costum convolution python kernel which includes the main part of ReSprop algorithm. 

### Training time

Please note that this code is showing the functionality of ReSprop algorithm and to gain the speedup a hardware accelerator is required.
