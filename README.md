lex Liu & Kevin Li, from NTU CSIE

---
## Desciprtion

This is our final project repository of the course ADLxMLDS 2017, Fall.

![](fig/fig_2.jpg)

In this project, we implement [FaderNet](https://arxiv.org/pdf/1706.00409.pdf) (NIPS 2017) and do capacity/reproducbility/ablation study. Our results can be find in the [poster](fig/post.pdf).

We've noticed that FaceBook had released [the offical github for FaderNet](https://github.com/facebookresearch/FaderNetworks). Since we've started the project slightly earlier than it's release, **ONLY in the part of testing FaderNet on unseen data (out of CelebA) had we used the model & testing code FaceBook released. For all the remaining parts including training & experiments, we're using our own production.**

## Dependency & Requirement

### Packages used in this project

Please make sure each of them is installed with the correct version

- numpy
- SciPy
- pytorch (0.3.0.post4)
- pandas (0.20.3)
- skimage (0.13.1)
- matplotlib (2.1.1)
- OpenCV
- Makefile

### Hardware Requirement

We're running our experiments with following hardware setting

- CPU : AMD Ryzen 5 1600 (6 cores 12 threads)
- GPU : GTX 1070 (with 8G memory)
- RAM : 16G (swap space on Disk is also 16G)
 
*** GPU with CUDA installed correctly ***
*** GPU memory MUST provide 8G free memory AT LEAST ***

## Training

TODO

## Testing
### Testing with our own code & model

The model we've trained can be find in [checkpoint/](checkpoint/). The testing images are the first 10 images in CelebA's testing set, we've uploaded the preprocessed version of them in [data/img/](data/img/) in order to let this part of testing can be done without downloading extra data/model

To generate [fig2](fig/fig_2.jpg) in the poster (Reproducibility Study in Experiments), run

        make reproduce

The result will be slightly better than the one in the poster since it's now using the model 100000 steps after the one we've used in poster.

To generate [fig3]() & [fig4]() in the poster (Ablation Study in Experiments), run

        make aga's code

aga's comments

### Testing with FaderNet's official release
