function net = initialize_qp1_fcn8s(fcn16s_model_path, num_classes)
 
net_ = load(fcn16s_model_path);
net = dagnn.DagNN.loadobj(net_.net);
net.removeLayer('deconv16');

%% add fcn layers
upsample_filters = single(bilinear_u(4, 1, num_classes)) ;
net.addLayer('deconv16_2', ...
             dagnn.ConvTranspose('size', size(upsample_filters), 'upsample', 2,...
                                 'crop', [1 1 1 1],'hasBias',false), ...
             'x54', 'x55', 'deconv16_2f') ;
net = initialize_param(net, 'deconv16_2f', upsample_filters, 0, 1);
% 
skip3 = dagnn.Conv('size',[1,1,256,num_classes], 'pad', 0);
net.addLayer('skip3', skip3, 'x24', 'x56', {'skip3f', 'skip3b'});
net = initialize_param(net, 'skip3f', zeros(1,1,256,num_classes), 0.01, 1);
net = initialize_param(net, 'skip3b', zeros(1,1,num_classes), 2, 1);
%
net.addLayer('sum8', dagnn.Sum(), {'x55', 'x56'}, 'x57');
% 
upsample_filters = single(bilinear_u(16, num_classes, num_classes)) ;
upsample_block = dagnn.ConvTranspose('size', size(upsample_filters),...
                                     'upsample', 8, 'crop', 4, ...
                                     'numGroups', num_classes, 'hasBias', false);
net.addLayer('deconv8', upsample_block, 'x57', 'prediction', 'deconv8f');
net = initialize_param(net, 'deconv8f', upsample_filters, 0, 1);

function net = initialize_param(net, name, value, learningRate, ...
                                weightDecay)
f = net.getParamIndex(name);
net.params(f).value = single(value);
net.params(f).learningRate = learningRate;
net.params(f).weightDecay = weightDecay;
