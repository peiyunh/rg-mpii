% CVPR 16: to show our results
%    run evaluatePCKh_val([1,2,5,7], 1)

function p = getExpParamsCVPR(predidx)

% path to the directory containg ground truth 'annolist' and RELEASE structures
p.gtDir = './ground_truth/';
p.colorIdxs = [1 1];

switch predidx
  case 0
    p.name = 'experiment name';
    % replicated RELEASE structure containing predictions on test images only
    % predictions are stored in the same way as GT body joint annotations
    % i.e. annolist_test = annolist(RELEASE.img_train == 0);
    p.predFilename = './predictions.mat';
    p.colorIdxs = [7 1];
  case 1
    p.name = 'qp2-4x';
    p.predFilename = 'cvpr_models/topdown-mpii-bot50-fcn-4s/result-epoch-10-val.mat';
  case 2
    p.name = 'qp1-4x';
    p.predFilename = 'cvpr_models/bottomup-mpii-bot50-fcn-4s/result-epoch-10-val.mat';
end

p.colorIdxs = [max(1,predidx) 1];
p.colorName = getColor(p.colorIdxs);
p.colorName = p.colorName ./ 255;

end
