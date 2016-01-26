=== Bottom-Up and Top-Down Reasoning with Hierarchical Rectified Gaussians ===

To run the code:

- Download [MatConvNet](http://www.vlfeat.org/matconvnet/) and put in matconvnet/
  - also have the pre-trained vgg-16 model at ./matconvnet/imagenet-vgg-verydeep-16.mat
- Download [MPII](http://human-pose.mpi-inf.mpg.de/) and put in data/mpii_human/
  - see data/mpii_human/README for more details
- Call run_qp1_mpii.m for training & testing a qp1 model
- Call run_qp2_mpii.m for training & testing a qp2 model

Credits:

The implementation is based on & inspired by [MatConvNet](http://www.vlfeat.org/matconvnet/) and
[MatConvNet-FCN](https://github.com/vlfeat/matconvnet-fcn). 

Other useful functions:

- dag_viz.m: Visualize a DAG-based model (Credit: [James](https://github.com/jsupancic/))
- memorySize.m: Estimate the memory usage of an object
