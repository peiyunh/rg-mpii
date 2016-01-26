% revised from eval/flatten_annolist.m

function [annolist_flat,single_person_flat, img_train_flat, ...
          img_index_flat, person_index_flat] = ...
        flatten_annotation(annolist,single_person, img_train, is_val)
    

annolist_flat = struct('image',[],'annorect',[]);

n = 0;
single_person_flat = [];
for imgidx = 1:length(annolist)
    rect_gt = annolist(imgidx).annorect;
    for ridx = 1:length(rect_gt)
        if (isfield(rect_gt(ridx),'objpos') && ~isempty(rect_gt(ridx).objpos))
            n = n + 1;
            annolist_flat(n).image.name = annolist(imgidx).image.name;
            annolist_flat(n).annorect = rect_gt(ridx);
            single_person_flat(n) = ismember(ridx,single_person{imgidx});
            img_index_flat(n) = imgidx;
            person_index_flat(n) = ridx;
            if ~isempty(is_val{imgidx}) && is_val{imgidx}(ridx) == 1
                img_train_flat(n) = 2;
            else
                img_train_flat(n) = img_train(imgidx);
            end
        end
    end
end

