% run this to
%    1. remark the train/val/test flags according to standard val
%    2. flatten the annotation to be person-based
%    3. unify how we define is_visible
clear all;

load('annotation/mpii_human_pose_v1_u12_1.mat');
res = load(['/sshd/peiyunh/workspace/fcn-key/baselines/' ...
            'mpii_valid_pred/data/detections.mat']);

RELEASE.img_train(RELEASE.img_train == 0) = 3;
RELEASE.img_test(RELEASE.img_train == 1) = 1;

is_val = cell(numel(RELEASE.annolist), 1);
% mark val persons 
for i = 1:numel(res.RELEASE_img_index)
    imgidx = res.RELEASE_img_index(i);
    ridx = res.RELEASE_person_index(i);
    % set val flag
    if isempty(is_val{imgidx})
        is_val{imgidx} = zeros(1, numel(RELEASE.annolist(imgidx).annorect));
    end
    is_val{imgidx}(ridx) = 1;
end

% flatten the annotation
[annolist, single_person, image_set, image_index, person_index] = flatten_annotation(...
    RELEASE.annolist, RELEASE.single_person, RELEASE.img_train, is_val);

%
image_name = cell(1, numel(annolist));
anno_rect = cell(1, numel(annolist));

% extract image paths out of annotation
for i = 1:numel(annolist)
    image_name{i} = annolist(i).image.name;
    anno_rect{i} = annolist(i).annorect;
    if isfield(anno_rect{i}, 'annopoints')
        points = anno_rect{i}.annopoints.point;
        if isfield(points, 'is_visible')
            for j = 1:numel(points)
                if isempty(points(j).is_visible)
                    points(j).is_visible = 1;
                elseif ischar(points(j).is_visible)
                    points(j).is_visible = ...
                        str2num(points(j).is_visible);
                elseif islogical(points(j).is_visible)
                    1;
                else
                    error('Cannot recognize the format');
                end
            end
        end
        anno_rect{i}.annopoints.point = points; 
    end
end

save('annotation/mpii_human_pose_remarked.mat', 'image_name', ...
     'anno_rect', 'single_person', 'image_set', 'image_index', 'person_index');
