function annolist = packResults(paths)

% for Nov-20-2015
%paths = {'models/topdown-mpii-fcn-4s/result-epoch-10-test.mat', ...
%         'models/topdown-mpii-fcn-8s/result-epoch-10-test.mat', ...
%         'models/topdown-mpii-fcn-16s/result-epoch-12-test.mat', ...
%         'models/bottomup-mpii-fcn-4s/result-epoch-10-test.mat', ...
%         'models/bottomup-mpii-fcn-8s/result-epoch-10-test.mat', ...
%         'models/bottomup-mpii-fcn-16s/result-epoch-10-test.mat',...
%         'models/bottomup-mpii-fcn-32s/result-epoch-30-test.mat',...
%        };

% for Jan-18-2016
paths = {'models/topdown-mpii-fcn4s/result-epoch-10-test.mat', ...
         'models/topdown-mpii-fcn8s/result-epoch-10-test.mat', ...
         'models/topdown-mpii-fcn16s/result-epoch-12-test.mat', ...
         'models/bottomup-mpii-fcn4s/result-epoch-10-test.mat', ...
         'models/bottomup-mpii-fcn8s/result-epoch-10-test.mat', ...
         'models/bottomup-mpii-fcn16s/result-epoch-10-test.mat',...
         'models/bottomup-mpii-fcn32s/result-epoch-30-test.mat',...
        };

         
for i = 1:numel(paths)
    path = paths{i};
    res = load(path);
    % re-unflatten 
    pred = re_unflatten(res);
    % save 
    [fdir,fname,ext] = fileparts(path);
    [~,dname,~] = fileparts(fdir);
    save(fullfile('results', [dname '-' fname]), 'pred');
end


function pred = re_unflatten(res)

cache_file = 'test_indices.mat';
if ~exist(cache_file)
    load('ground_truth/mpii_human_pose_v1_u12_1.mat');
    test_idx = find(RELEASE.img_train==0);
    test_inv_idx = -ones(size(test_idx));
    for i = 1:numel(test_idx)
        test_inv_idx(test_idx(i)) = i;
    end
    save(cache_file, 'test_idx', 'test_inv_idx');
else
    load(cache_file);
end

for i = 1:numel(test_idx)
    pred_idx = find(res.image_index == test_idx(i));
    for j = 1:numel(pred_idx)
        pred(i).annorect(j) = res.pred(pred_idx(j)).annorect; 
    end
end
