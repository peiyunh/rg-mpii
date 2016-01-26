% CVPR 16: to show our results
%    run evaluatePCKh_val([1,2,5,7], 1)

function p = getExpParams(predidx)

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
    %p.predFilename = 'models/topdown-mpii-fcn-4s/result-epoch-9-val.mat';
    p.predFilename = 'models/qp2-fcn4s/result-epoch-3-val.mat';
  case 2
    p.name = 'qp1-4x';
    %p.predFilename = 'models/bottomup-mpii-fcn-4s/result-epoch-8-val.mat';
    p.predFilename = 'models/qp1-fcn4s/result-epoch-1-val.mat';
  case 3
    p.name = 'qp2-8x';
    p.predFilename = 'models/qp2-fcn8s/result-epoch-1-val.mat';
  case 4
    p.name = 'qp1-8x';
    p.predFilename = 'models/qp1-fcn8s/result-epoch-1-val.mat';
  case 5
    p.name = 'qp2-16x';
    p.predFilename = 'models/qp2-fcn16s/result-epoch-10-val.mat';
  case 6
    p.name = 'qp1-16x';
    p.predFilename = 'models/qp1-fcn16s/result-epoch-9-val.mat';
  case 7
    p.name = 'qp1-32x';
    p.predFilename = 'models/qp1-fcn32s/result-epoch-25-val.mat';
  case 8
    p.name = 'qp3-4x-notrain';
    p.predFilename = 'models/qp3-fcn4s-notrain/result-epoch-3-val.mat';
  case 9
    p.name = 'qp3-4x';
    p.predFilename = 'models/qp3-fcn4s/result-epoch-6-val.mat';
  case 10
    p.name = 'qp4-4x-notrain';
    p.predFilename = 'models/qp4-fcn4s-notrain/result-epoch-3-val.mat';
    %case 6
    %p.name = 'qp1-16x';
    %p.predFilename = 'models/bottomup-mpii-fcn-16s/result-epoch-6-val.mat';
    %p.predFilename = 'models/bottomup-mpii-fcn-16s/result-epoch-10-val.mat';
    %case 7
    %p.name = 'qp-32x';
    %p.predFilename =
    %'models/bottomup-mpii-fcn-32s/result-epoch-29-val.mat';
    %p.predFilename = 'models/bottomup-mpii-fcn-32s/result-epoch-10-val.mat';
end

p.colorIdxs = [max(1,predidx) 1];
%p.colorName = getColor(p.colorIdxs);
p.colorName = rand(1,3);
p.colorName = p.colorName ./ 255;

end
