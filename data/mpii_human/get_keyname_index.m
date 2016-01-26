function [KeyName2Idx, Idx2KeyName] = get_keyname_index()

KeyName2Idx = containers.Map();
Idx2KeyName = {'r-ankle', 'r-knee', 'r-hip', 'l-hip', 'l-knee', ...
               'l-ankle', 'pelvis', 'thorax', 'upper-neck', ...
               'head-top', 'r-wrist', 'r-elbow', 'r-shoulder', ...
               'l-shoulder', 'l-elbow', 'l-wrist'};
for i = 1:numel(Idx2KeyName)
    KeyName2Idx(Idx2KeyName{i}) = i;
end

