function prepare_data_bot50()
global KeyName2Idx
global Idx2KeyName
global base_size

% here shows what bot50 means, which is include 50 more at bottom
base_size = [100, 100, 100, 150];

% VARS: image_name, anno_rect, single_person, image_set
load('annotation/mpii_human_pose_remarked.mat');

[KeyName2Idx, Idx2KeyName] = get_keyname_index();

train_idx = find(image_set == 1);
val_idx = find(image_set == 2);
test_idx = find(image_set == 3);

prepare(image_name(train_idx), anno_rect(train_idx), ...
        single_person(train_idx), image_set(train_idx), ...
        image_index(train_idx), person_index(train_idx), 'train');
prepare(image_name(val_idx), anno_rect(val_idx), ...
        single_person(val_idx), image_set(val_idx), ...
        image_index(val_idx), person_index(val_idx), 'val');
prepare(image_name(test_idx), anno_rect(test_idx), ...
        single_person(test_idx), image_set(test_idx), ...
        image_index(test_idx), person_index(test_idx), 'test');

function create_if_not_exist(folder)
if ~exist(folder)
    mkdir(folder);
end

function prepare(image_names, anno_rects, single_persons, ...
                 image_sets, image_index, person_index, dataset_name)

global KeyName2Idx
global Idx2KeyName
global base_size

create_if_not_exist(fullfile('images', dataset_name));
create_if_not_exist(fullfile('labels', dataset_name));

OBJECT_SIZE = [224,224];
IMAGE_SIZE = [256,256];
BORDER_SIZE = (IMAGE_SIZE-OBJECT_SIZE)/2;

for i = 1:numel(image_names)
    fprintf('Preparing data for %s: %d/%d\n', dataset_name, i, ...
            numel(image_names));

    image_path = sprintf('images/%s/%08d.jpg', dataset_name, i);
    label_path = sprintf('labels/%s/%08d.mat', dataset_name, i);

    im = imread(fullfile('full_images', image_names{i}));

    objpos = anno_rects{i}.objpos;
    scale = anno_rects{i}.scale;

    label = struct('image_index', image_index(i), ...
                   'person_index', person_index(i));

    bx1 = objpos.x - scale * base_size(1);
    by1 = objpos.y - scale * base_size(2);
    bx2 = objpos.x + scale * base_size(3);
    by2 = objpos.y + scale * base_size(4);
    
    label.anchor = [bx1,by1,bx2,by2];
    
    spl = bx1 - 1; spr = size(im,2) - bx2;
    spt = by1 - 1; spb = size(im,1) - by2;
    sfs = OBJECT_SIZE ./ [by2-by1+1, bx2-bx1+1];
    pad = [max(ceil(BORDER_SIZE./sfs - [spt,spl]), 0);
           max(ceil(BORDER_SIZE./sfs - [spb,spr]), 0)];

    if any(pad(1,:))
        im = padarray(im, pad(1,:), 0, 'pre');
    end
    if any(pad(2,:))
        im = padarray(im, pad(2,:), 0, 'post');
    end
    
    nbx1 = floor(bx1-BORDER_SIZE(2)./sfs(2))+pad(1,2); 
    nbx2 = floor(bx2+BORDER_SIZE(2)./sfs(2))+pad(1,2);
    nby1 = ceil(by1-BORDER_SIZE(1)./sfs(1))+pad(1,1);
    nby2 = ceil(by2+BORDER_SIZE(1)./sfs(1))+pad(1,1);
    
    im = im(nby1:nby2, nbx1:nbx2, :);
    im = imresize(im, IMAGE_SIZE, 'bicubic');

    obj_x = sfs(2) * (objpos.x + pad(1,2) - nbx1 + 1);
    obj_y = sfs(1) * (objpos.y + pad(1,1) - nby1 + 1);
    label.obj = [obj_x, obj_y];
    label.single = single_persons(i);

    if image_sets(i) ~= 3 % if in test set 
        hx1 = anno_rects{i}.x1;
        hy1 = anno_rects{i}.y1;
        hx2 = anno_rects{i}.x2;
        hy2 = anno_rects{i}.y2;

        points = anno_rects{i}.annopoints.point;

        kxs = [points.x];
        kys = [points.y];
        if isfield(points, 'is_visible')
            kvs = [points.is_visible];
        else
            kvs = zeros(size(kxs));
        end
        assert(numel(kvs) == numel(kxs));
        kids = [points.id] + 1; % index from 1

        kys = sfs(1) * (kys + pad(1,1) - nby1 + 1);
        kxs = sfs] (2) * (kxs + pad(1,2) - nbx1 + 1);

        hy1 = sfs(1) * (hy1 + pad(1,1) - nby1 + 1);
        hy2 = sfs(1) * (hy2 + pad(1,1) - nby1 + 1);
        hx1 = sfs(2) * (hx1 + pad(1,2) - nbx1 + 1);
        hx2 = sfs(2) * (hx2 + pad(1,2) - nbx1 + 1);
        
        label.x = kxs;
        label.y = kys;
        label.v = kvs;
        label.id = kids;
        label.head = [hx1,hy1,hx2,hy2];
    end

    if ~exist(image_path), imwrite(im, image_path); end
    imwrite(im, image_path); 
    save(label_path, '-struct', 'label');

end
