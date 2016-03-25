Bottom-Up and Top-Down Reasoning with Hierarchical Rectified Gaussians =====

**Instructions**:

- Compile MatConvNet (If you use CuDNN, make sure you set the $PATH and
  $LD_LIBRARY_PATH correct)
- Download the pre-trained vgg-16 model from
  [MatConvNet](http://www.vlfeat.org/matconvnet/) and put in matconvnet/
- Download [MPII](http://human-pose.mpi-inf.mpg.de/) and put in data/mpii_human/
  - see data/mpii_human/README for more details
- Call run_qp1_mpii.m for training & testing a qp1 model
- Call run_qp2_mpii.m for training & testing a qp2 model


**Pretrained models**:

Download our trained models
[here](http://www.ics.uci.edu/~peiyunh/public/rg-mpii/).


**Notes**:

A few necessary changes are made based on the original
[MatConvNet](http://www.vlfeat.org/matconvnet/).

See [our code and models](http://www.ics.uci.edu/~peiyunh/public/rg-mpii/) for
face landmark localization on [AFLW](https://lrs.icg.tugraz.at/research/aflw/)
dataset.


**Credits**:

The implementation is based on & inspired by
[MatConvNet](http://www.vlfeat.org/matconvnet/) and
[MatConvNet-FCN](https://github.com/vlfeat/matconvnet-fcn).

Thanks [James](https://github.com/jsupancic/) for dag_viz.m which dumps a
MatConvNet model into a dot file.

