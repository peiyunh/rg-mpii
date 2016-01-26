function [images, labels] = cnn_get_batch_mpii(image_paths, label_paths, varargin)
% data augmentation:
% - up to 32px random shifts 
% - random left/right flipping

opts.alpha = 0.5;
opts.imageSize = [227, 227];
opts.labelSize = [227, 227];
opts.border = [29, 29] ;
opts.keepAspect = true ;
opts.numAugments = 1 ;
opts.transformation = 'none' ;
opts.averageImage = [] ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts = vl_argparse(opts, varargin);

opts.labelSize = double(opts.labelSize);
opts.imageSize = double(opts.imageSize);

%% data set specific parameters
SC_BIAS = 0.6 ;
AVG_HEAD_SIZE = 90; % pixels

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = numel(image_paths) >= 1 && ischar(image_paths{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

if prefetch
  vl_imreadjpeg(image_paths, 'numThreads', opts.numThreads, 'prefetch') ;
  images = [];
  labels = [];
  return ;
end
if fetch
  im = vl_imreadjpeg(image_paths,'numThreads', opts.numThreads) ;
else
  im = image_paths ;
end

tfs = [] ;
switch opts.transformation
  case 'none'
    tfs = [
      .5 ;
      .5 ;
       0 ] ;
  case 'f5'
    tfs = [...
      .5 0 0 1 1 .5 0 0 1 1 ;
      .5 0 1 0 1 .5 0 1 0 1 ;
       0 0 0 0 0  1 1 1 1 1] ;
  case 'f25'
    [tx,ty] = meshgrid(linspace(0,1,5)) ;
    tfs = [tx(:)' ; ty(:)' ; zeros(1,numel(tx))] ;
    tfs_ = tfs ;
    tfs_(3,:) = 1 ;
    tfs = [tfs,tfs_] ;
  case 'stretch'
  otherwise
    error('Uknown transformations %s', opts.transformation) ;
end
[~,transformations] = sort(rand(size(tfs,2), numel(image_paths)), 1) ;

if ~isempty(opts.rgbVariance) && isempty(opts.averageImage)
  opts.averageImage = zeros(1,1,3) ;
end
if numel(opts.averageImage) == 3
  opts.averageImage = reshape(opts.averageImage, 1,1,3) ;
end

images = zeros(opts.imageSize(1), opts.imageSize(2), 3, ...
               numel(image_paths)*opts.numAugments, 'single');
% make label the same size as the images
labels = -ones(opts.labelSize(1), opts.labelSize(2), opts.labelSize(3), ...
               numel(image_paths)*opts.numAugments, 'single');

si = 1 ;
for i=1:numel(image_paths)

  % acquire image
  if isempty(im{i})
    imt = imread(image_paths{i}) ;
    imt = single(imt) ; % faster than im2single (and multiplies by 255)
  else
    imt = im{i} ;
  end
  if size(imt,3) == 1
    imt = cat(3, imt, imt, imt) ;
  end

  % resize
  w = size(imt,2) ;
  h = size(imt,1) ;
  factor = [(opts.imageSize(1)+opts.border(1))/h ...
            (opts.imageSize(2)+opts.border(2))/w];

  if opts.keepAspect
    factor = max(factor) ;
  end
  if any(abs(factor - 1) > 0.0001)
    imt = imresize(imt, ...
                   'scale', factor, ...
                   'method', opts.interpolation) ;
  end
  
  label = load(label_paths{i});
  
  % crop & flip
  w = size(imt,2) ;
  h = size(imt,1) ;
  for ai = 1:opts.numAugments
    switch opts.transformation
      case 'stretch'
        sz = round(min(opts.imageSize(1:2)' .* (1-0.1+0.2*rand(2,1)), [w;h])) ;
        dx = randi(w - sz(2) + 1, 1) ;
        dy = randi(h - sz(1) + 1, 1) ;
        flip = rand > 0.5 ;
      otherwise
        tf = tfs(:, transformations(mod(ai-1, numel(transformations)) + 1)) ;
        sz = opts.imageSize(1:2) ;
        dx = floor((w - sz(2)) * tf(2)) + 1 ;
        dy = floor((h - sz(1)) * tf(1)) + 1 ;
        flip = tf(3) ;
    end

    sx = round(linspace(dx, sz(2)+dx-1, opts.imageSize(2))) ;
    sy = round(linspace(dy, sz(1)+dy-1, opts.imageSize(1))) ;
    if flip, sx = fliplr(sx) ; end

    if ~isempty(opts.averageImage)
      offset = opts.averageImage ;
      if ~isempty(opts.rgbVariance)
        offset = bsxfun(@plus, offset, reshape(opts.rgbVariance * randn(3,1), 1,1,3)) ;
      end
      images(:,:,:,si) = bsxfun(@minus, imt(sy,sx,:), offset) ;
    else
      images(:,:,:,si) = imt(sy,sx,:) ;
    end

    % compute annotation
    if ~isfield(label, 'x') || ~isfield(label, 'y')
        si = si + 1; continue; 
    end
    lx = label.x - dx + 1;
    ly = label.y - dy + 1;
    lhx = label.head([1,3]) - dx + 1;
    lhy = label.head([2,4]) - dy + 1;

    loc_lv = (1 <= lx & lx <= opts.imageSize(2) & ...
              1 <= ly & ly <= opts.imageSize(1));

    if flip,
        lx = opts.imageSize(2) - lx + 1;
        lhx([2,1]) = opts.imageSize(2) - lhx([1,2]) + 1;
        lid = flip_mpii_annotation(label.id);
    else
        lid = label.id;
    end
    
    head_size = norm(label.head(3:4)-label.head(1:2));

    loc_heat = -ones(opts.labelSize(1:2));
    for j = 1:numel(lx)
        loc_heat(:) = -1;
        [xx,yy] = meshgrid(1:opts.labelSize(2), ...
                           1:opts.labelSize(1));
        radius = opts.alpha * SC_BIAS * head_size;
        I = (sqrt((yy-ly(j)).^2 + (xx-lx(j)).^2) <= radius);
        loc_heat(I) = 1;
        labels(:,:,lid(j),si) = loc_heat;
    end
    si = si + 1 ;
  end
end

function fid = flip_mpii_annotation(id)
map = [6,5,4,3,2,1,...
       7,8,9,10,...
       16,15,14,13,12,11];
fid = map(id);
