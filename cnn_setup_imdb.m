% cnn_setup_data: 
% function imdb = cnn_setup_imdb(data, opts)
% imdb:
%  -inputSize 
%  -labelSize 
%  -images
%     -id
%     -name 
%     -set 
%     -label

function imdb = cnn_setup_imdb(opts)
rng(0);

imdb.imageDir = fullfile(opts.dataDir, 'images');
imdb.labelDir = fullfile(opts.dataDir, 'labels');

imdb.classes.KeyName2Idx = get_keyname_index();

imdb.imageSize = [256,224,3];
imdb.labelSize = [256,224,imdb.classes.KeyName2Idx.Count];

% train/val 
names = {};
labels = {};
for d = dir(fullfile(opts.dataDir, 'images/train/*.jpg'))'
    names{end+1} = ['train/' d.name];
    labels{end+1} = ['train/' strrep(d.name, 'jpg', 'mat')];
end
num = numel(names);
imdb.images.id = 1:num;
imdb.images.set = ones(1, num);
imdb.images.name = names; 
imdb.images.label = labels;

% randomly split 20% for validation
if ~exist(fullfile(opts.dataDir, 'images/val'))
    imdb.images.set(randperm(num, round(0.2*num))) = 2;
else
    names = {};
    labels = {};
    for d = dir(fullfile(opts.dataDir, 'images/val/*.jpg'))'
        names{end+1} = ['val/' d.name];
        labels{end+1} = ['val/' strrep(d.name, 'jpg', 'mat')];
    end
    num = numel(names);
    imdb.images.id = horzcat(imdb.images.id, (1:num) + 1e7);
    imdb.images.set = horzcat(imdb.images.set, 2*ones(1, num));
    imdb.images.name = horzcat(imdb.images.name, names);
    imdb.images.label = horzcat(imdb.images.label, labels);
end

% test
names = {};
labels = {};
for d = dir(fullfile(opts.dataDir, 'images/test/*.jpg'))'
    names{end+1} = ['test/' d.name];
    labels{end+1} = ['test/' strrep(d.name, 'jpg', 'mat')];
end
num = numel(names);
imdb.images.id = horzcat(imdb.images.id, (1:num) + 2e7);
imdb.images.set = horzcat(imdb.images.set, 3*ones(1, num));
imdb.images.name = horzcat(imdb.images.name, names);
imdb.images.label = horzcat(imdb.images.label, labels);
