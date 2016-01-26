function cnn_qp1_mpii(varargin)
global VOCopts

startup; 

opts.dataDir = fullfile('data/mpii_human') ;
opts.modelType = 'fcn32s';
opts.networkType = 'dagnn';
opts.alpha = 0.5;
[opts, varargin] = vl_argparse(opts, varargin) ;
 
addpath(opts.dataDir);

opts.expDir = fullfile('models', ['qp1-' opts.modelType]);
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.gpus = [1];
opts.batchSize = 40;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 4 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.momentum = 0.90;
opts.train.batchSize = opts.batchSize;
opts.train.numSubBatches = 1;
opts.train.continue = true ;
opts.train.gpus = opts.gpus ;
opts.train.prefetch = true ;
opts.train.sync = false ; 
opts.train.cudnn = true ;
opts.train.expDir = opts.expDir ;

opts.learningRate = 1e-6;
[opts, varargin] = vl_argparse(opts, varargin) ;
opts.train.learningRate = opts.learningRate;

opts.numEpochs = 50;
opts = vl_argparse(opts, varargin);
opts.train.numEpochs = opts.numEpochs;

if ~exist(opts.expDir), mkdir(opts.expDir); end;

switch opts.modelType
  case 'fcn32s'
    % ------------------------------------------------------------
    %% Load imdb, otherwise setup imdb 
    if exist(opts.imdbPath)
        imdb = load(opts.imdbPath);
        fprintf('Loaded imdb from %s: size %f MB.\n', ...
                opts.imdbPath, memorySize(imdb));
    else
        imdb = cnn_setup_imdb(opts);
        fprintf('Saving imdb into %s: size %f MB\n', ...
                opts.imdbPath, memorySize(imdb));
        save(opts.imdbPath, '-struct', 'imdb', '-v7.3');
    end
    assert(imdb.labelSize(3) == 16);
    
    net = initialize_fcn32s(double(imdb.labelSize(3)));
    net.meta.normalization.imageSize = imdb.imageSize;
    net.meta.normalization.labelSize = imdb.labelSize;
    %net.meta.normalization.border = 256 -
    %net.meta.normalization.imageSize(1:2);
    net.meta.normalization.border = [36, 32];
    
    net.meta.normalization.interpolation = 'bicubic';
    net.meta.normalization.averageImage = [];
    net.meta.normalization.keepAspect = true;

    bopts = net.meta.normalization;
    bopts.numThreads = opts.numFetchThreads;
    bopts.numAugments = 1;
    bopts.transformation = 'f25';
    
    % compute image statistics
    imageStatsPath = fullfile(opts.expDir, 'imageStats.mat');
    if exist(imageStatsPath)
        load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    else
        [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
        save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    end
    % 
    net.meta.normalization.averageImage = rgbMean ;
    %    
    [v,d] = eig(rgbCovariance) ;
    bopts.transformation = 'f25' ;
    bopts.averageImage = rgbMean ;
    bopts.rgbVariance = 0.1*sqrt(d)*v' ;
    useGpu = numel(opts.train.gpus) > 0 ;
  case 'fcn16s'
    imdbPath = strrep(opts.imdbPath, '16s', '32s');
    imdb = load(imdbPath);
    assert(imdb.labelSize(3) == 16);
    fcn32s_dir = strrep(opts.expDir, '16s', '32s');
    fcn32s_epoch = findLastCheckpoint(30, fcn32s_dir);
    fcn32s_model = sprintf('net-epoch-%d.mat', fcn32s_epoch);
    fcn32s_model = fullfile(fcn32s_dir, fcn32s_model);
    fprintf('Training fcn16s based on %s\n', fcn32s_model);
    %
    net = initialize_qp1_fcn16s(fcn32s_model, double(imdb.labelSize(3)));
    % compute image statistics
    imageStatsPath = fullfile(opts.expDir, 'imageStats.mat');
    imageStatsPath = strrep(imageStatsPath, '16s', '32s');
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    %
    bopts = net.meta.normalization;
    [v,d] = eig(rgbCovariance) ;
    bopts.numThreads = opts.numFetchThreads;
    bopts.numAugments = 1;
    bopts.transformation = 'f25' ;
    bopts.averageImage = rgbMean ;
    bopts.rgbVariance = 0.1*sqrt(d)*v' ;

    useGpu = numel(opts.train.gpus) > 0 ;
  case 'fcn8s'
    imdbPath = strrep(opts.imdbPath, '8s', '32s');
    imdb = load(imdbPath);
    assert(imdb.labelSize(3) == 16);
    fcn16s_dir = strrep(opts.expDir, '8s', '16s');
    fcn16s_epoch = findLastCheckpoint(15, fcn16s_dir);
    fcn16s_model = sprintf('net-epoch-%d.mat', fcn16s_epoch);
    fcn16s_model = fullfile(fcn16s_dir, fcn16s_model);
    fprintf('Training fcn8s based on %s\n', fcn16s_model);
    %
    net = initialize_qp1_fcn8s(fcn16s_model, double(imdb.labelSize(3)));
    % compute image statistics
    imageStatsPath = fullfile(opts.expDir, 'imageStats.mat');
    imageStatsPath = strrep(imageStatsPath, '8s', '32s');
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    %
    bopts = net.meta.normalization;
    [v,d] = eig(rgbCovariance) ;
    bopts.numThreads = opts.numFetchThreads;
    bopts.numAugments = 1;
    bopts.transformation = 'f25' ;
    bopts.averageImage = rgbMean ;
    bopts.rgbVariance = 0.1*sqrt(d)*v' ;

    useGpu = numel(opts.train.gpus) > 0 ;
  case 'fcn4s'
    imdbPath = strrep(opts.imdbPath, '4s', '32s');
    imdb = load(imdbPath);
    assert(imdb.labelSize(3) == 16);
    fcn8s_dir = strrep(opts.expDir, '4s', '8s');
    fcn8s_epoch = findLastCheckpoint(100, fcn8s_dir);
    fcn8s_model = sprintf('net-epoch-%d.mat', fcn8s_epoch);
    fcn8s_model = fullfile(fcn8s_dir, fcn8s_model);
    fprintf('Training fcn4s based on %s\n', fcn8s_model);
    %
    net = initialize_qp1_fcn4s(fcn8s_model, double(imdb.labelSize(3)));
    % compute image statistics
    imageStatsPath = fullfile(opts.expDir, 'imageStats.mat');
    imageStatsPath = strrep(imageStatsPath, '4s', '32s');
    load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
    %
    bopts = net.meta.normalization;
    [v,d] = eig(rgbCovariance) ;
    bopts.numThreads = opts.numFetchThreads;
    bopts.numAugments = 1;
    bopts.transformation = 'f25' ;
    bopts.averageImage = rgbMean ;
    bopts.rgbVariance = 0.1*sqrt(d)*v' ;
    useGpu = numel(opts.train.gpus) > 0 ;
end

% setup diary file 
diary_file = sprintf([datestr(now,'HH-MM-SS.mm-dd-yy') '.log']);
diary(fullfile(opts.expDir, diary_file));

% 
opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
% setup different get batch functions for train/val
bopts.alpha = opts.alpha;
fcn = cell(1,2);
bopts.numAugments = 1;
fn{1} = getBatchDagNNWrapper(bopts, useGpu) ;
bopts.transformation = 'none';
bopts.rgbVariance = [];
bopts.numAugments = 1;
fn{2} = getBatchDagNNWrapper(bopts, useGpu) ;

%
opts
bopts
opts.train

info = cnn_train_dag(net, imdb, fn, opts.train) ;

% test, and with a much larger batch size
opts.test = opts.train;
opts.test.batchSize = ceil(1.8*opts.train.batchSize);
fn{2} = getBatchDagNNWrapper(bopts, useGpu);
cnn_test_mpii(net, imdb, fn{2}, opts.test);

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [images,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
image_paths = strcat([imdb.imageDir filesep], ...
                     imdb.images.name(batch));
label_paths = strcat([imdb.labelDir filesep], ...
                     imdb.images.label(batch));
[images, labels] = cnn_get_batch_mpii(image_paths, label_paths, opts,...
                                                  'prefetch', nargout == 0) ;
% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
image_paths = strcat([imdb.imageDir filesep], ...
                     imdb.images.name(batch)) ;
label_paths = strcat([imdb.labelDir filesep], ...
                     imdb.images.label(batch));
[images, labels] = cnn_get_batch_mpii(image_paths, label_paths, opts,...
                                                  'prefetch', nargout == 0) ;

if nargout > 0
    if useGpu
        images = gpuArray(images) ;
    end
    inputs = {'input', images, 'label', labels};
end
% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 10: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
    z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
    n = size(z,2) ;
    avg{t} = mean(temp, 4) ;
    rgbm1{t} = sum(z,2)/n ;
    rgbm2{t} = z*z'/n ;
    batch_time = toc(batch_time) ;
    fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;

% --------------------------------------------------------------------
function filter = init_upsample_filter(sz)
% --------------------------------------------------------------------
% Credits: 
%    Jon https://gist.github.com/shelhamer/80667189b218ad570e82#file-solve-py
f = floor((sz+1)/2); 
if mod(sz,2) == 1
    c = f - 1;
else
    c = f - 0.5; 
end
[x,y] = meshgrid(1:sz,1:sz); 
filter = (1-abs(x-c-1)/f) .* (1-abs(y-c-1)/f);

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(numEpochs, modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([min(epoch, numEpochs) 0]) ;
