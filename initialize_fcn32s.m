function net = initialize_fcn32s(num_classes)

% load pretrained net and build index
pretrain_net = load('matconvnet/imagenet-vgg-verydeep-16.mat');
pretrain_idx = containers.Map();
for i = 1:numel(pretrain_net.layers)
    pretrain_idx(pretrain_net.layers{i}.name) = i;
end

% create a network which has the position of pool and relu switched
net = cnn_imagenet_init('model', 'vgg-vd-16', 'batchNormalization', true);
net.layers(end) = [];
net.layers = net.layers([1:5, 7,6, 8:end]);        % pool1, relu1_2
net.layers = net.layers([1:12, 14,13, 15:end]);    % pool2, relu2_2
net.layers = net.layers([1:22, 24,23, 25:end]);    % pool3, relu3_3
net.layers = net.layers([1:32, 34,33, 35:end]);    % pool4, relu4_3
net.layers = net.layers([1:42, 44,43, 45:end]);    % pool5, relu5_3

% fill in weights from pretrained 
for i = 1:numel(net.layers)
    layer_name = net.layers{i}.name;
    if ~isKey(pretrain_idx, layer_name), continue; end;
    net.layers{i} = pretrain_net.layers{pretrain_idx(layer_name)};
end

% initialize pre-trained layers learning rate factors
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true);
for i = 1:numel(net.layers)
    if (isa(net.layers(i).block, 'dagnn.Conv') && ...
        net.layers(i).block.hasBias)
        filt = net.getParamIndex(net.layers(i).params{1});
        bias = net.getParamIndex(net.layers(i).params{2});
        net.params(bias).learningRate = 2 * ...
            net.params(filt).learningRate ;
    end
end
net.removeLayer('fc8');

% pad fc6
net.layers(45).block.pad = [3,3,3,3];

% create predictors
%filter32 = dagnn.ConvTranspose('size',[7,7,num_classes,4096], 'crop', ...
%                               0, 'hasBias', true);
filter32 = dagnn.Conv('size', [1,1,4096,num_classes], 'hasBias', true); 
net.addLayer('filter32', filter32, 'x50', 'x51', ...
             {'filter32f', 'filter32b'});
net = initialize_param(net, 'filter32f', zeros(1,1,4096,num_classes), 1, 1);
net = initialize_param(net, 'filter32b', zeros(1,num_classes), 2, 1);

%
upsample_filters = single(bilinear_u(64, num_classes, num_classes)) ;
upsample_block = dagnn.ConvTranspose('size', size(upsample_filters),...
                                     'upsample', 32, 'crop', 16, ...
                                     'numGroups', num_classes, 'hasBias', false);
net.addLayer('deconv32', upsample_block, 'x51', 'prediction', 'deconv32f');
net = initialize_param(net, 'deconv32f', upsample_filters, 0, 1);

%
net.addLayer('objective', dagnn.Loss('loss', 'logistic'), ...
             {'prediction', 'label'}, 'objective');

function net = initialize_param(net, name, value, learningRate, ...
                                weightDecay)
f = net.getParamIndex(name);
net.params(f).value = single(value);
net.params(f).learningRate = learningRate;
net.params(f).weightDecay = weightDecay;
