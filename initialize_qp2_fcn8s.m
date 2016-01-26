function net = initialize_qp2_fcn8s(fcn16s_model_path, num_classes)
 y 
net_ = load(fcn16s_model_path);
net = dagnn.DagNN.loadobj(net_.net);
net.removeLayer('deconv16');

%% add top down layers first
unpool4 = dagnn.UnPooling('poolSize', [2,2], 'stride', [2,2]);
net.addLayer('unpool4', unpool4, {'x34_sum_relu', 'x31'}, 'unpool4');

% 
deconv4_3 = dagnn.ConvTranspose('size',[3,3,512,512],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv4_3', deconv4_3, 'unpool4', 'x29_t', 'conv4_3f');
net.addLayer('bnorm_x29t', dagnn.BatchNorm(), ...
             'x29_t', 'x29_tbn', {'bnorm_x29tf', 'bnorm_x29tb', 'bnorm_x29tm'});
net = initialize_param(net, 'bnorm_x29tf', ones(512,1), 1, 1);
net = initialize_param(net, 'bnorm_x29tb', 0.001*ones(512,1), 2, 1);
net = initialize_param(net, 'bnorm_x29tm', zeros(512,2), 0.05, 0);
net.addLayer('sum_x30', dagnn.Sum(), {'x29', 'x29_tbn'}, 'x30_sum')
net.addLayer('relu_sum_x30', dagnn.ReLU(), 'x30_sum', 'x30_sum_relu');

deconv4_2 = dagnn.ConvTranspose('size',[3,3,512,512],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv4_2', deconv4_2, 'x30_sum_relu', 'x26_t', 'conv4_2f');
net.addLayer('bnorm_x26t', dagnn.BatchNorm(), ...
             'x26_t', 'x26_tbn', {'bnorm_x26tf', 'bnorm_x26tb', 'bnorm_x26tm'});
net = initialize_param(net, 'bnorm_x26tf', ones(512,1), 1, 1);
net = initialize_param(net, 'bnorm_x26tb', 0.001*ones(512,1), 2, 1);
net = initialize_param(net, 'bnorm_x26tm', zeros(512,2), 0.05, 0);
net.addLayer('sum_x27', dagnn.Sum(), {'x26', 'x26_tbn'}, 'x27_sum')
net.addLayer('relu_sum_x27', dagnn.ReLU(), 'x27_sum', 'x27_sum_relu');

deconv4_1 = dagnn.ConvTranspose('size',[3,3,256,512],'upsample',1, ...
                                'crop',1,'hasBias',false);
net.addLayer('deconv4_1', deconv4_1, 'x27_sum_relu', 'x23_t', 'conv4_1f');
net.addLayer('bnorm_x23t', dagnn.BatchNorm(), ...
             'x23_t', 'x23_tbn', {'bnorm_x23tf', 'bnorm_x23tb', 'bnorm_x23tm'});
net = initialize_param(net, 'bnorm_x23tf', ones(256,1), 1, 1);
net = initialize_param(net, 'bnorm_x23tb', 0.001*ones(256,2), 2, 1);
net = initialize_param(net, 'bnorm_x23tm', zeros(256,2), 0.05, 0);
net.addLayer('sum_x24', dagnn.Sum(), {'x23', 'x23_tbn'}, 'x24_sum')
net.addLayer('relu_sum_x24', dagnn.ReLU(), 'x24_sum', 'x24_sum_relu');

%% add fcn layers
upsample_filters = single(bilinear_u(4, 1, num_classes)) ;
net.addLayer('deconv16_2', ...
             dagnn.ConvTranspose('size', size(upsample_filters), 'upsample', 2,...
                                 'crop', [1 1 1 1],'hasBias',false), ...
             'x54', 'x55', 'deconv16_2f') ;
net = initialize_param(net, 'deconv16_2f', upsample_filters, 0, 1);
% 
skip3 = dagnn.Conv('size',[1,1,256,num_classes], 'pad', 0);
net.addLayer('skip3', skip3, 'x24_sum_relu', 'x56', {'skip3f', 'skip3b'});
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
