function net = initialize_qp2_fcn4s(fcn8s_model_path, num_classes)

net_ = load(fcn8s_model_path);
net = dagnn.DagNN.loadobj(net_.net);
net.removeLayer('deconv8');

%% add top down layers first
unpool3 = dagnn.UnPooling('poolSize', [2,2], 'stride', [2,2]);
net.addLayer('unpool3', unpool3, {'x24_sum_relu', 'x21'}, 'unpool3');

% 
deconv3_3 = dagnn.ConvTranspose('size',[3,3,256,256],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv3_3', deconv3_3, 'unpool3', 'x19_t', 'conv3_3f');
net.addLayer('bnorm_x19t', dagnn.BatchNorm(), ...
             'x19_t', 'x19_tbn', {'bnorm_x19tf', 'bnorm_x19tb', 'bnorm_x19tm'});
net = initialize_param(net, 'bnorm_x19tf', ones(256,1), 1, 1);
net = initialize_param(net, 'bnorm_x19tb', 0.001*ones(256,1), 2, 1);
net = initialize_param(net, 'bnorm_x19tm', zeros(256,2), 0.05, 0);
net.addLayer('sum_x20', dagnn.Sum(), {'x19', 'x19_tbn'}, 'x20_sum')
net.addLayer('relu_sum_x20', dagnn.ReLU(), 'x20_sum', 'x20_sum_relu');

deconv3_2 = dagnn.ConvTranspose('size',[3,3,256,256],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv3_2', deconv3_2, 'x20_sum_relu', 'x16_t', 'conv3_2f');
net.addLayer('bnorm_x16t', dagnn.BatchNorm(), ...
             'x16_t', 'x16_tbn', {'bnorm_x16tf', 'bnorm_x16tb', 'bnorm_x16tm'});
net = initialize_param(net, 'bnorm_x16tf', ones(256,1), 1, 1);
net = initialize_param(net, 'bnorm_x16tb', 0.001*ones(256,1), 2, 1);
net = initialize_param(net, 'bnorm_x16tm', zeros(256,2), 0.05, 0);
net.addLayer('sum_x17', dagnn.Sum(), {'x16', 'x16_tbn'}, 'x17_sum')
net.addLayer('relu_sum_x17', dagnn.ReLU(), 'x17_sum', 'x17_sum_relu');

deconv3_1 = dagnn.ConvTranspose('size',[3,3,128,256],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv3_1', deconv3_1, 'x17_sum_relu', 'x13_t', 'conv3_1f');
net.addLayer('bnorm_x13t', dagnn.BatchNorm(), ...
             'x13_t', 'x13_tbn', {'bnorm_x13tf', 'bnorm_x13tb', 'bnorm_x13tm'});
net = initialize_param(net, 'bnorm_x13tf', ones(128,1), 1, 1);
net = initialize_param(net, 'bnorm_x13tb', 0.001*ones(128,1), 2, 1);
net = initialize_param(net, 'bnorm_x13tm', zeros(128,2), 0.05, 0);
net.addLayer('sum_x14', dagnn.Sum(), {'x13', 'x13_tbn'}, 'x14_sum')
net.addLayer('relu_sum_x14', dagnn.ReLU(), 'x14_sum', 'x14_sum_relu');

%% add fcn layers
upsample_filters = single(bilinear_u(4, 1, num_classes)) ;
net.addLayer('deconv8_2', ...
             dagnn.ConvTranspose('size', size(upsample_filters), 'upsample', 2,...
                                 'crop', [1 1 1 1],'hasBias',false), ...
             'x57', 'x58', 'deconv8_2f') ;
net = initialize_param(net, 'deconv8_2f', upsample_filters, 0, 1);
% 
skip2 = dagnn.Conv('size',[1,1,256,num_classes], 'pad', 0);
net.addLayer('skip2', skip2, 'x14_sum_relu', 'x59', {'skip2f', 'skip2b'});
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
