function annolist = packResults(path)

paths = {'models/topdown-mpii-fcn-4s/result-epoch-10-test.mat', ...
         'models/topdown-mpii-fcn-8s/result-epoch-10-test.mat', ...
         'models/topdown-mpii-fcn-16s/result-epoch-12-test.mat', ...
         'models/bottomup-mpii-fcn-4s/result-epoch-10-test.mat', ...
         'models/bottomup-mpii-fcn-8s/result-epoch-10-test.mat', ...
         'models/bottomup-mpii-fcn-16s/result-epoch-10-test.mat',...
         'models/bottomup-mpii-fcn-32s/result-epoch-30-test.mat',...
        };

for i = 1:numel(paths)
    path = paths{i};
    res = load(path);
    %
    annolist = struct('annorect', {});

    %
    for i = 1:numel(res.image_index)
        ii = res.image_index(i); % image index
        annolist(ii) = res.pred(i);
    end

    % squeeze
    idx = [];
    for i = 1:numel(annolist)
        if isempty(annolist(i).annorect),
            idx(end+1) = i;
        end
    end
    annolist(idx) = [];

    % rename it  
    pred = annolist;

    [fdir,fname,ext] = fileparts(path);
    [~,dname,~] = fileparts(fdir);


    %
    if ~exist(fullfile('results', dname))
        mkdir(fullfile('results', dname));
    end

    %
    copyfile(path, fullfile('results', dname, ['flatten-' fname '.mat']));
    save(fullfile('results', dname, ['unflatten-' fname]), 'pred');
end
