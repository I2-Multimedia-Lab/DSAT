# DSAT
The code of ' Degradation-Aware Self-Attention Based Transformer for Blind Image Super-Resolution '.

# Abstract
Compared to CNN-based methods, Transformer-based methods achieve impressive image restoration performance due to their ability to model remote dependencies. However, how to apply the Transformer-based method to the field of blind super-resolution (SR) and further make the SR network adaptive to degradation information is still an open question. In this paper, we proposed a degradation-aware self-attention-based Transformer model, where we incorporate contrastive learning into the Transformer for the learning of the degradation representations of the input images with unknown noises. Especially, we combine CNN and Transformer in the SR network, where we first use the CNN modulated by the degradation information to extract local features, and then employ degradation-aware Transformer to extract global semantic features. We apply our method to several popular large-scale benchmark datasets for testing, and achieve state-of-the-art performance compared to existing methods. In particular, our method yields a PSNR of 32.43 dB on the Urban100 dataset at  $\times$2 scale,  0.94 dB higher than  DASR, and 26.62 db on the Urban100 dataset at $\times$4 scale,  0.26 dB improvement over KDSR, setting a new record in this area.

# Requirements
Pytorch == 1.12.1
torchvision == 0.13.1
opencv-python
tensorboardX
einops
skimage
numpy

# Train Data Preparation
1. Prepare training data
1.1 Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and the [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) dataset.
1.2 Combine the HR images from these two datasets in your_data_path/DF2K/HR to build the DF2K dataset

# Test Data Preparation
Download [benchmark datasets](https://github.com/xinntao/BasicSR/blob/a19aac61b277f64be050cef7fe578a121d944a0e/docs/Datasets.md) (e.g., Set5, Set14 and other test sets) and prepare HR/LR images in your_data_path/benchmark.

# Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/sanghyun-son/EDSR-PyTorch), [DASR](https://github.com/The-Learning-And-Vision-Atelier-LAVA/DASR) and [MoCo](https://github.com/facebookresearch/moco). We thank the authors for sharing the codes.

