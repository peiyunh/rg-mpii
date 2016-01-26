function net = initialize_qp1_fcn4s(fcn8s_model_path, num_classes)
 
net_ = load(fcn8s_model_path);
net = dagnn.DagNN.loadobj(net_.net);
net.removeLayer('deconv8');

%% add fcn layers
upsample_filters = single(bilinear_u(4, 1, num_classes)) ;
net.addLayer('deconv8_2', ...
             dagnn.ConvTranspose('size', size(upsample_filters), 'upsample', 2,...
                                 'crop', [1 1 1 1],'hasBias',false), ...
             'x57', 'x58', 'deconv8_2f') ;
net = initialize_param(net, 'deconv8_2f', upsample_filters, 0, 1);
% 
skip3 = dagnn.Conv('size',[1,1,256,num_classes], 'pad', 0);
net.addLayer('skip2', skip3, 'x14', 'x59', {'skip2f', 'skip2b'});
net = initialize_param(net, 'skip2f', zeros(1,1,128,num_classes), 0.001, 1);
net = initialize_param(net, 'skip2b', zeros(1,1,num_classes), 2, 1);
%
net.addLayer('sum4', dagnn.Sum(), {'x58', 'x59'}, 'x60');
% 
upsample_filters = single(bilinear_u(8, num_classes, num_classes)) ;
upsample_block = dagnn.ConvTranspose('size', size(upsample_filters),...
                                     'upsample', 4, 'crop', 2, ...
                                     'numGroups', num_classes, 'hasBias', false);
net.addLayer('deconv4', upsample_block, 'x60', 'prediction', 'deconv4f');
net = initialize_param(net, 'deconv4f', upsample_filters, 0, 1);

function net = initialize_param(net, name, value, learningRate, ...
                                weightDecay)
f = net.getParamIndex(name);
net.params(f).value = single(value);
net.params(f).learningRate = learningRate;
net.params(f).weightDecay = weightDecay;
