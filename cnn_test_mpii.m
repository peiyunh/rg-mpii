function stats = cnn_test_mpii(net, imdb, getBatch, varargin)

opts.expDir = fullfile('data','exp') ;
opts.continue = false ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.derOutputs = {'objective', 1} ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.extractStatsFn = @extractStats ;
opts = vl_argparse(opts, varargin) ;

SC_BIAS = 0.6;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelPath = modelPath(opts.numEpochs);
resPath = @(ep) fullfile(opts.expDir, sprintf('result-epoch-%d.mat', ep));
resPath = resPath(opts.numEpochs);

%
%testset = 'test';
testset = 'val';
%testset = 'test-small';
if strcmp(testset, 'val')
    opts.test = find(imdb.images.set==2);
    resPath = strrep(resPath, '.mat', '-val.mat');
elseif strcmp(testset, 'test')
    opts.test = find(imdb.images.set==3);
    resPath = strrep(resPath, '.mat', '-test.mat');
elseif strcmp(testset, 'test-small')
    opts.test = find(imdb.images.set==3);
    rng(1);
    opts.test = opts.test(randsample(numel(opts.test), 300));
    resPath = strrep(resPath, '.mat', '-testsmall-1.mat');
end

net_ = load(modelPath);
net = dagnn.DagNN.loadobj(net_.net);

if numel(opts.gpus) >= 1
    net.move('gpu');
end
net.addLayer('sigmoid', dagnn.Sigmoid(), 'prediction', 'score');
net.mode = 'test';

[KeyName2Idx, Idx2KeyName] = get_keyname_index();
KeyError = nan(numel(Idx2KeyName), numel(opts.test));

result_dir = strrep(opts.expDir, 'models', 'results');
if ~exist(result_dir)
    mkdir(result_dir);
end

% define output
pred_image_index = [];
pred_person_index = [];
pred_annolist = [];

% define heat map we want to see
%heatmaps = zeros([imdb.labelSize, numel(opts.test)], 'single');

%
bopts = net.meta.normalization;
bopts.imageSize(1:2) = [256,224];
bopts.labelSize(1:2) = [256,224];

num = 0;
fprop_costs = [];
startTime = tic;
for t=1:opts.batchSize:numel(opts.test)
    batchSize = min(opts.batchSize, numel(opts.test) - t + 1);
    batchStart = t;
    batchEnd = min(t + batchSize - 1, numel(opts.test));
    batch = opts.test(batchStart:batchEnd);
    inputs = getBatch(imdb, batch);

    num = num + numel(batch);
    if numel(batch) == 0, continue; end;

    if opts.prefetch
        batchStart = t + numel(batch);
        batchEnd = min(t + 2*numel(batch) - 1, numel(opts.test));
        nextBatch = opts.test(batchStart:batchEnd);
        getBatch(imdb, nextBatch);
    end

    %tim = tic; 
    net.eval(inputs);
    %fprop_costs(end+1) = toc(tim) ./ opts.batchSize;
    %fprintf('Time cost: %fs\n', mean(fprop_costs));
    
    scores = gather(net.vars(net.getVarIndex('score')).value);

    %vis_scores = scores(:,:,1:end/2,:);
    %loc_scores = scores(:,:,end/2+1:end,:);
    loc_scores = scores;
    label_paths = strcat([imdb.labelDir, filesep], ...
                         imdb.images.label(batch));

    %heatmaps(:,:,:,t:t+numel(batch)-1) = scores;
    for i = 1:numel(batch)
        label = load(label_paths{i});

        xys = nan(2, numel(Idx2KeyName));
        vvs = nan(1, numel(Idx2KeyName));
        for j = 1:numel(Idx2KeyName)
            [pV,pI] = max(reshape(loc_scores(:,:,j,i),[],1));
            [py,px] = ind2sub(bopts.imageSize(1:2),pI);            
            xys(:,j) = [px;py];
            %vvs(j) = vis_scores(py,px,j,i);
            vvs(j) = loc_scores(py,px,j,i);
        end
        % transform to original coordinate system
        ax1 = label.anchor(1); ay1 = label.anchor(2);
        ax2 = label.anchor(3); ay2 = label.anchor(4);
        rev_sc = [(ax2 - ax1 + 1) / bopts.imageSize(2);
                  (ay2 - ay1 + 1) / bopts.imageSize(1)];
        xys = bsxfun(@plus,bsxfun(@times,xys,rev_sc),[ax1;ay1]);

        % capsule the prediction result
        point = struct();
        for j = 1:numel(Idx2KeyName);
            point(j).id = j - 1;
            point(j).x = xys(1,j);
            point(j).y = xys(2,j);
            point(j).is_visible = vvs(j);
        end
        pred_image_index(t+i-1) = label.image_index;
        pred_person_index(t+i-1) = label.person_index;
        pred_annolist(t+i-1).annorect.annopoints.point = point;
    end        
    fprintf('Test model of epoch %d: %.2f seconds, %d/%d\n', opts.numEpochs, ...
            toc(startTime), t+numel(batch)-1, numel(opts.test));
end

%save(resPath, 'KeyError', 'stats');
%save(resPath, 'KeyError', 'stats', 'keypointsAll',
%'output_joints');
image_index = pred_image_index;
person_index = pred_person_index;
pred = pred_annolist;

%save(resPath, 'pred', 'image_index', 'person_index', 'heatmaps',
%'-v7.3');
save(resPath, 'pred', 'image_index', 'person_index', '-v7.3');
