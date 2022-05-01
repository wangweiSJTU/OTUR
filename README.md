# Optimal Transport for Unsupervised Denoising Learning - _Official Pyorch implementation of the TPAMI 2022 paper_
**Wei Wang\***, **Fei Wen\***, **Zeyu Yan\***, **Peilin Liu**

**Abstract**:
_Recently, much progress has been made in unsupervised denoising learning. However, existing methods more or less rely on some assumptions on the signal and/or degradation model, which limits their practical performance. How to construct an optimal criterion for unsupervised denoising learning without any prior knowledge on the degradation model is still an open question. Toward answering this question, this work proposes a criterion for unsupervised denoising learning based on the optimal transport theory. This criterion has favorable properties, e.g., approximately maximal preservation of the information of the signal, whilst achieving perceptual reconstruction. Furthermore, though a relaxed unconstrained formulation is used in practical implementation, we prove that the relaxed formulation in theory has the same solution as the original constrained formulation. Experiments on synthetic and real-world data, including realistic photographic, microscopy, depth, and raw depth images, demonstrate that the proposed method even compares favorably with supervised methods, e.g., approaching the PSNR of supervised methods while having better perceptual quality. Particularly, for spatially correlated noise and realistic microscopy images, the proposed method not only achieves better perceptual
quality but also has higher PSNR than supervised methods. Besides, it shows remarkable superiority in harsh practical conditions with complex noise, e.g., raw depth images._

The official publication in IEEE TPAMI is available at: https://ieeexplore.ieee.org/document/9763342.

The trained models are provided in the `./checkpoint` folder of each experiment. The proposed formulation is implemented in an adversarial training framework using [WGAN-gp](https://proceedings.neurips.cc/paper/2017/hash/892c3b1c6dccd52936e27cbd0ff683d6-Abstract.html). The generator is modified from part of [MPRNet](https://github.com/swz30/MPRNet) and the discriminator is modified from that of  [SRGAN](https://github.com/tensorlayer/srgan). We use them here only for academic use purpose.

## Datasets

In this section, we will introduce the datasets we use for each experiment. 

### Synthetic Noisy RGB Images

For three kinds of noises: Gaussian noise, Poisson noise and brown Gaussian, we use a RGB image dataset [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) for training and the dataset [KODAK](http://r0k.us/graphics/kodak/) for testing. Training and testing code can be found in folder `./denoise_gaussian_rgb`, `./denoise_poisson_rgb` and `./denoise_brown_rgb`.

### Synthetic Noisy Depth Images

We use a synthetic depth image dataset [SUNCG](https://sscnet.cs.princeton.edu/) for training and the depth image dataset [Middlebury](https://vision.middlebury.edu/stereo/data/) for testing. Training and testing code can be found in folder `./denoise_gaussian_depth`.

### Real-world Microscope Images

We use a real fluorescence microscopy image dataset [FMD](https://drive.google.com/drive/folders/1aygMzSDdoq63IqSk-ly8cMq0_owup8UM) for training and testing. Training and testing code can be found in folder `./denoise_micro`.

### Real-World Photographic Images

We use a real smartphone photographic image dataset [SIDD](http://www.cs.yorku.ca/~kamel/sidd/) for training and testing. Training and testing code can be found in folder `./denoise_photographic`.

### Real-World Depth Images

For training our model, we use a real depth image dataset collected by a kinect camera [NYU](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html#raw_parts)
as the noisy sample domain, and randomly collect the same amount of patches from the SUNCG dataset as the clean target domain. The NYU dataset contains 1,449 depth images of indoor scenes and the corresponding RGB images captured by a Kinect camera. Training and testing code can be found in folder `./denoise_real_depth`.

### Real-World Raw Depth Images

As the Kinect camera only provides pre-processed images, we additionally consider a raw depth denoising experiment using another commercial ToF camera which provides raw depth data without any pre-processing. We collect 1,430 raw depth images using this camera as the noisy sample domain, which can be find at https://drive.google.com/file/d/14DcXk1mBW_RkgEYEb_eIhdoX78TjCtUk/view?usp=sharing. For the clean target domain, we also randomly collect the same amount of patches from the SUNCG dataset. Training and testing code can be found in folder `./denoise_real_depth`.

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
python train.py --nEpochs=200 --noise_sigma=50 --gpus="0" --trainset="../../BSDS500/" --sigma=30
```

To train with the N2C method:

```
cd denoise_gaussian_rgb
python train_n2c.py --nEpochs=200 --noise_sigma=50 --gpus="0" --trainset="../../BSDS500/"
```

To train with the N2N method:

```
cd denoise_gaussian_rgb
python train_n2n.py --nEpochs=200 --noise_sigma=50 --gpus="0" --trainset="../../BSDS500/"
```

### Validation using a trained network

Take the synthetic Gaussian noise denoising on RGB images as an example, to run a validation dataset through a trained network:

```
cd denoise_gaussian_rgb
python test.py --model="./checkpoint/model_denoise_unet_n2c50200.pth" --dataset="./KODAK" --save="./results" --noise_sigma=25 --gpu="0"
```

## Some results

### (1) Results on synthetic noisy images:

**Visual comparison on synthetic noisy images with brown Gaussian noise. The PSNR/PI/LPIPS results are provided in the brackets. The images are enlarged for clarity.**

![alt text](images/brown_gaussian.png )

**Quantitative distortion comparison (PSNR/SSIM) on synthetic noisy RGB images with Gaussian, Poisson and Brown Gaussian noise.**

![alt text](images/rgb_denoise.png )

**Quantitative perceptual quality comparison (PI/LPIPS) on synthetic noisy RGB images with Gaussian, Poisson and Brown Gaussian noise.**

![alt text](images/rgb_denoise_pi.png )


### (2) Results on real-world microscope images:

**Visual comparison on real-world microscopy images, where the ground-truth is obtained by averaging over 50 realizations of each scene. The PSNR/PI/LPIPS results are provided in the brackets. The images are enlarged for clarity.**

![alt text](images/micro.png )

**Quantitative comparison on real-world microscope images.:**

![alt text](images/micro_denoise.png )

### (3) Results on real-world raw depth images:

**Visual comparison on a real-world raw depth image captured by a commercial ToF camera. The RGB image is only used as a reference of the scene, which is not aligned with the depth image. “OT denoising” denotes our model trained on synthetic depth images with Gaussian noise.**

![alt text](images/raw.png )

## Citation

If this work helps you, please consider citing:

    @ARTICLE{OTUR,
        author={Wang, Wei and Wen, Fei and Yan, Zeyu and Liu, Peilin},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
        title={Optimal Transport for Unsupervised Denoising Learning}, 
        year={2022},
        pages={1-1},
        doi={10.1109/TPAMI.2022.3170155}
    }
