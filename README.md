# MuFIN
This is the PyTorch implementation of paper 'Learning from Text: A Multimodal Face Inpainting Network for Irregular Holes', which can be found [here](https://ieeexplore.ieee.org/document/10445705/).
### Introduction:
We propose a single-stage Multimodal Face Inpainting Network (MuFIN) that fills the irregular holes based on both the contextual information of the input face image and the provided text information. To fully exploit the rest parts of the corrupted face images, a plug-and-play Multi-scale Multi-level Skip Fusion Module (MMSFM), which extracts multi-scale features and fuses shallow features into deep features at multiple levels, is illustrated. Moreover, to bridge the gap between textual and visual modalities and effectively fuse cross-modal features, a Multi-scale Text-Image Fusion Block (MTIFB), which incorporates text features into image features from both local and global scales, is developed. 
![framework](https://raw.githubusercontent.com/FVL2020/MuFIN/main/figs/framework.png)  
# Prerequisites
* Python 3.7
* Pytorch 1.7
* NVIDIA GPU + CUDA cuDNN
# Installation
* Clone this repo:  
```
git clone https://github.com/FVL2020/MuFIN
cd MuFIN-master
```
* Install Pytorch
* Install python requirements:
```
pip install -r requirements.txt
```
# Preparation
Run scripts/flist.py to generate train, test and validation set file lists. For example, to generate the training set file list on CelebA dataset, you should run:  
```
python ./scripts/flist.py --path path_to_celebA_train_set --output ./datasets/celeba_train.flist
```
# Training
Run:
```
python train.py --checkpoints [path to checkpoints]
```
# Testing
Run:
```
python test.py --checkpoints [path to checkpoints]
```
# Evaluating
Run:
```
python ./scripts/metrics.py --data-path [path to ground truth] --output-path [path to model output]
```
Then run the "read_data.m" file to obtain PSNR, SSIM and Mean Absolute Error under different mask ratios. The "log_metrics.dat" and "log_test.dat" files are in the [output-path].   
To measure the Fréchet Inception Distance (FID score), run:
```
python ./scripts/fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]
```
# Results
### Quantitative Results:
![quantitative_results1](https://raw.githubusercontent.com/FVL2020/MuFIN/main/figs/quantitative_results1.png)  
Quantitative comparisons of our MuFIN with state-of-the-art CNN-based methods on CelebA and Multi-Modal-CelebA-HQ datasets.
![quantitative_results2](https://raw.githubusercontent.com/FVL2020/MuFIN/main/figs/quantitative_results2.png)  
Quantitative comparisons of our MuFIN with state-of-the-art Transformer-based and Diffusion Model-based methods on Multi-Modal-CelebA-HQ dataset.
### Qualitative Results:
![qualitative_results1](https://raw.githubusercontent.com/FVL2020/MuFIN/main/figs/qualitative_results1.png)  
Qualitative comparisons of our MuFIN with state-of-the-art CNN-based methods on CelebA and Multi-Modal-CelebA-HQ datasets.
![qualitative_results2](https://raw.githubusercontent.com/FVL2020/MuFIN/main/figs/qualitative_results2.png)  
Qualitative comparisons of our MuFIN with state-of-the-art Transformer-based and Diffusion Model-based methods on Multi-Modal-CelebA-HQ dataset.
# Citation
Please cite us if you find this work helps.  
```
@inproceedings{MuFIN,
  title={Learning from Text: A Multimodal Face Inpainting Network for Irregular Holes},
  author={Zhan, Dandan and Wu, Jiahao and Luo, Xing and Jin, Zhi},
  booktitle={TCSVT},
  year={2024},
}
```
# Appreciation
The codes refer to EdgeConnect. Thanks for the authors of it!
# License
This repository is released under the MIT License as found in the LICENSE file. Code in this repo is for non-commercial use only.
