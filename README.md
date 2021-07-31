# Unsupervised Image Restoration Based on Optimal Transport

This code is used to reproduce the results of the optimal transport based restoration algortihm in the paper: 

Wei Wang, Zeyu Yan, Fei Wen, Rendong Ying, and Peilin Liu, "Optimal Transport for Unsupervised Restoration
Learning". 

Trained models are also provided in the `./checkpoint` folder. The proposed formulation is implemented in an adversarial training framework using [WGAN-gp](https://proceedings.neurips.cc/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html). The generator is modified from part of [MPRNet](https://github.com/swz30/MPRNet) and the discriminator is modified from that of  [SRGAN](https://github.com/tensorlayer/srgan). We use them here only for academic use purpose.

## Datasets

In this section, we will introduce the datasets we use for each experiment. 

### Synthetic Noisy RGB Images

For three kinds of noises: Gaussian noise, Poisson noise and brown Gaussian, we use a RGB image dataset [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) for training and the dataset [KODAK](http://r0k.us/graphics/kodak/) for testing.

### Synthetic Noisy Depth Images

We use a synthetic depth image dataset [SUNCG](https://sscnet.cs.princeton.edu/) for training and the depth image dataset [Middlebury](https://vision.middlebury.edu/stereo/data/) for testing.

### Real-world Microscope Images

We use a real fluorescence microscopy image dataset [FMD](https://drive.google.com/drive/folders/1aygMzSDdoq63IqSk-ly8cMq0_owup8UM) for training and testing.

### Real-World Photographic Images

We use a real smartphone photographic image dataset [SIDD](http://www.cs.yorku.ca/~kamel/sidd/) for training and testing.

### Real-World Depth Images

For training our model, we use a real depth image dataset collected by a kinect camera [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html#raw_parts)
as the noisy sample domain, and randomly collect the same
amount of patches from the SUNCG dataset as the
clean target domain. The NYU dataset contains 1,449 depth
images of indoor scenes and the corresponding RGB images
captured by a Kinect camera.

### Real-World Raw Depth Images

As the Kinect camera only provides pre-processed images, we additionally consider a raw depth denoising experiment using another commercial ToF camera which provides raw depth data without any pre-processing. We collect 1,430 raw depth images using this camera as the noisy sample
domain, which can be find at . For the clean target domain, we also randomly collect the same amount of patches from the SUNCG dataset.

All these datasets are only used for academic purpose.

## Getting started

### Python requirements

This code requires:

- Python 3.6
- torch 1.8.0
- h5py, opencv-python, numpy

### Preparing training dataset

We use the hdf5 files to store training data, you can use the code `generate_train_compress.m` for RGB images and `generate_train_compress_depth.m` for depth images in the folder `/gdata` to generate your own dataset from image folders. And the code `readH5.py` is used to merge two hdf5 files into one, which is useful in some experiments.

### Training networks

Take the synthetic Gaussian noise denoising on RGB images as an example, to train the proposed method:

```
cd denoise_gaussian_rgb
python train.py --nEpochs=200 --noise_sigma=50 --gpus="0" --trainset="../../BSDS500" --sigma=30
```




Moreover, for ease of use for interested readers who want to reproduce the result of our algorithm,
and only for academic use purpose,
we have copied here the blurred images from the following two datasets (see the 'BlurryImages' and 'Levin_data' folders):
(1) R. Kohler, M. Hirsch, B. J. Mohler, B. Scholkopf, and S. Harmeling, “Recording and playback of camera shake: Benchmarking blind deconvolution with a real-world database,” in Proc. Eur. Conf. Comput. Vis., 2012, pp. 27–40.
(2) A. Levin, Y. Weiss, F. Durand, and W. T. Freeman, “Understanding and evaluating blind deconvolution algorithms,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2009, pp. 1964–1971.

By this, the results on these two datasets reported in the paper can be reproduced via directly running the 'demo_Levin.m' and ‘demo_eccv12.m’ files.


Meanwhile, some sample images form Pan are also used here, please see the 'sample_images' folder.

## (1) Results on the dataset of Kohler et al.:

**Quantitative evaluation results on the benchmark dataset of Kohler et al. (PSNR and SSIM comparison over 48 blurry images)**

<img src="https://github.com/FWen/deblur-pmp/blob/master/results_eccv12/Kohler_PSNR_SSIM.png" width="600" /> 

**Average PSNR and average SSIM on the dataset of Kohler et al.**

<img src="https://github.com/FWen/deblur-pmp/blob/master/results_eccv12/Kohler_PSNR_SSIM_table.png" width="300" />


## (2) Results on the dataset of Levin et al.:

**Quantitative evaluation results on the benchmark dataset of Levin et al. [2] (PSNR and SSIM comparison over 32 blurry images)**

<img src="https://github.com/FWen/deblur-pmp/blob/master/results_Levin/Levin_PSNR_SSIM.png?raw=true" width="500" />

**Average PSNR and average SSIM on the dataset of Levin et al.:**

<img src="https://github.com/FWen/deblur-pmp/blob/master/results_Levin/Levin_PSNR_SSIM_table.png?raw=true" width="300" />

## (3) Computational complexity:

<img src="https://github.com/FWen/deblur-pmp/blob/master/results_samples/comp/runtime.png" width="500" />
