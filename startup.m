%addpath tools;
addpath export_fig;

addpath matconvnet; 
addpath matconvnet/matlab;
addpath matconvnet/examples;
addpath matconvnet/examples/imagenet; 
vl_setupnn;

addpath VOCdevkit/VOCcode;
VOCinit;

dbstop if error;
