function net = initialize_qp1_fcn16s(fcn32s_model_path, num_classes)
 
net_ = load(fcn32s_model_path);
net = dagnn.DagNN.loadobj(net_.net);
net.removeLayer('deconv32');

%% add fcn layers
upsample_filters = single(bilinear_u(4, 1, num_classes)) ;
net.addLayer('deconv32_2', ...
             dagnn.ConvTranspose('size', size(upsample_filters), 'upsample', 2,...
                                 'crop', [1 1 1 1],'hasBias',false), ...
             'x51', 'x52', 'deconv32_2f') ;
net = initialize_param(net, 'deconv32_2f', upsample_filters, 0, 1);
% 
skip4 = dagnn.Conv('size',[1,1,512,num_classes], 'pad', 0);
net.addLayer('skip4', skip4, 'x34', 'x53', {'skip4f', 'skip4b'});
net = initialize_param(net, 'skip4f', zeros(1,1,512,num_classes), 0.1, 1);
net = initialize_param(net, 'skip4b', zeros(1,1,num_classes), 2, 1);
%
net.addLayer('sum16', dagnn.Sum(), {'x52', 'x53'}, 'x54');
% 
upsample_filters = single(bilinear_u(32, num_classes, num_classes)) ;
upsample_block = dagnn.ConvTranspose('size', size(upsample_filters),...
                                     'upsample', 16, 'crop', 8, ...
                                     'numGroups', num_classes, 'hasBias', false);
net.addLayer('deconv16', upsample_block, 'x54', 'prediction', 'deconv16f');
net = initialize_param(net, 'deconv16f', upsample_filters, 0, 1);


function net = initialize_param(net, name, value, learningRate, ...
                                weightDecay)
f = net.getParamIndex(name);
net.params(f).value = single(value);
net.params(f).learningRate = learningRate;
net.params(f).weightDecay = weightDecay;
