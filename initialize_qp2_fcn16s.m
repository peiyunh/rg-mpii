function net = initialize_qp2_fcn16s(fcn32s_model_path, num_classes)

net_ = load(fcn32s_model_path);
net = dagnn.DagNN.loadobj(net_.net);
net.removeLayer('deconv32');

%% add top down layers first
deconv_fc7 = dagnn.ConvTranspose('size',[1,1,4096,4096],'upsample',1,...
                                 'crop',0,'hasBias',false);
net.addLayer('deconv_fc7', deconv_fc7, 'x50', 'x46_t', 'fc7f');

net.addLayer('bnorm_x46t', dagnn.BatchNorm(), ...
             'x46_t', 'x46_tbn', {'bnorm_x46tf', 'bnorm_x46tb', 'bnorm_x46tm'});
net = initialize_param(net, 'bnorm_x46tf', ones(4096,1), 1, 1);
net = initialize_param(net, 'bnorm_x46tb', 0.001*ones(4096,1), 2, 1);
net = initialize_param(net, 'bnorm_x46tm', zeros(4096,2), 0.05, 0);

net.addLayer('sum_x47', dagnn.Sum(), {'x46', 'x46_tbn'}, 'x47_sum')
net.addLayer('relu_sum_x47', dagnn.ReLU(), 'x47_sum', 'x47_sum_relu');

%
deconv_fc6 = dagnn.ConvTranspose('size',[7,7,512,4096],'upsample',1, ...
                                 'crop',3,'hasBias',false);
net.addLayer('deconv_fc6', deconv_fc6, 'x47_sum_relu', 'x43_t', 'fc6f');

net.addLayer('bnorm_x43t', dagnn.BatchNorm(), ...
             'x43_t', 'x43_tbn', {'bnorm_x43tf', 'bnorm_x43tb', 'bnorm_x43tm'});
net = initialize_param(net, 'bnorm_x43tf', ones(512,1), 1, 1);
net = initialize_param(net, 'bnorm_x43tb', 0.001*ones(512,1), 2, 1);
net = initialize_param(net, 'bnorm_x43tm', zeros(512,2), 0.05, 0);

net.addLayer('sum_x44', dagnn.Sum(), {'x43', 'x43_tbn'}, 'x44_sum')
net.addLayer('relu_sum_x44', dagnn.ReLU(), 'x44_sum', 'x44_sum_relu');

%
unpool5 = dagnn.UnPooling('poolSize', [2,2], 'stride', [2,2]);
net.addLayer('unpool5', unpool5, {'x44_sum_relu', 'x41'}, 'unpool5');

% 
deconv5_3 = dagnn.ConvTranspose('size',[3,3,512,512],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv5_3', deconv5_3, 'unpool5', 'x39_t', 'conv5_3f');

net.addLayer('bnorm_x39t', dagnn.BatchNorm(), ...
             'x39_t', 'x39_tbn', {'bnorm_x39tf', 'bnorm_x39tb', 'bnorm_x39tm'});
net = initialize_param(net, 'bnorm_x39tf', ones(512,1), 1, 1);
net = initialize_param(net, 'bnorm_x39tb', 0.001*ones(512,1), 2, 1);
net = initialize_param(net, 'bnorm_x39tm', zeros(512,2), 0.05, 0);
net.addLayer('sum_x40', dagnn.Sum(), {'x39', 'x39_tbn'}, 'x40_sum')
net.addLayer('relu_sum_x40', dagnn.ReLU(), 'x40_sum', 'x40_sum_relu');

deconv5_2 = dagnn.ConvTranspose('size',[3,3,512,512],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv5_2', deconv5_2, 'x40_sum_relu', 'x36_t', 'conv5_2f');

net.addLayer('bnorm_x36t', dagnn.BatchNorm(), ...
             'x36_t', 'x36_tbn', {'bnorm_x36tf', 'bnorm_x36tb', 'bnorm_x36tm'});
net = initialize_param(net, 'bnorm_x36tf', ones(512,1), 1, 1);
net = initialize_param(net, 'bnorm_x36tb', 0.001*ones(512,1), 2, 1);
net = initialize_param(net, 'bnorm_x36tm', zeros(512,2), 0.05, 0);
net.addLayer('sum_x37', dagnn.Sum(), {'x36', 'x36_tbn'}, 'x37_sum')
net.addLayer('relu_sum_x37', dagnn.ReLU(), 'x37_sum', 'x37_sum_relu');

deconv5_1 = dagnn.ConvTranspose('size',[3,3,512,512],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv5_1', deconv5_1, 'x37_sum_relu', 'x33_t', 'conv5_1f');
net.addLayer('bnorm_x33t', dagnn.BatchNorm(), ...
             'x33_t', 'x33_tbn', {'bnorm_x33tf', 'bnorm_x33tb', 'bnorm_x33tm'});
net = initialize_param(net, 'bnorm_x33tf', ones(512,1), 1, 1);
net = initialize_param(net, 'bnorm_x33tb', 0.001*ones(512,1), 2, 1);
net = initialize_param(net, 'bnorm_x33tm', zeros(512,2), 0.05, 0);
net.addLayer('sum_x34', dagnn.Sum(), {'x33', 'x33_tbn'}, 'x34_sum')
net.addLayer('relu_sum_x34', dagnn.ReLU(), 'x34_sum', 'x34_sum_relu');

%% add fcn layers
upsample_filters = single(bilinear_u(4, 1, num_classes)) ;
net.addLayer('deconv32_2', ...
             dagnn.ConvTranspose('size', size(upsample_filters), 'upsample', 2,...
                                 'crop', [1 1 1 1],'hasBias',false), ...
             'x51', 'x52', 'deconv32_2f') ;
net = initialize_param(net, 'deconv32_2f', upsample_filters, 0, 1);
% 
skip4 = dagnn.Conv('size',[1,1,512,num_classes], 'pad', 0);
net.addLayer('skip4', skip4, 'x34_sum_relu', 'x53', {'skip4f', 'skip4b'});
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
