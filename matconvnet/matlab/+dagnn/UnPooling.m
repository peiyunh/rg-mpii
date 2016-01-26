% inputs: {top-down, bottom-up}
classdef UnPooling < dagnn.Filter
  properties
    method = 'max'
    poolSize = [1 1]
    opts = {'cuDNN'}
    argmax = []
  end

  methods
    function outputs = forward(self, inputs, params)
    % compute argmax from inputs{2} (using backprop of pooling)
      onemap = ones(size(inputs{1}),'single');
      if isa(inputs{1}, 'gpuArray') || isa(inputs{2}, 'gpuArray')
          onemap = gpuArray(onemap);
      end
      self.argmax = vl_nnpool(inputs{2}, self.poolSize, onemap, ...
                         'pad', self.pad, 'stride', self.stride, ...
                         'method', self.method, self.opts{:});
      % apply argmax to upsample inputs{1}
      outputs{1} = vl_nnpool(self.argmax, self.poolSize, inputs{1}, ...
                             'pad', self.pad, 'stride', self.stride, ...
                             'method', self.method);
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      % use argmax as mask for average pooling
      mask = self.argmax * prod(self.poolSize); % to cancel the average
      derInputs{1} = vl_nnpool(derOutputs{1}.*mask, self.poolSize, ...
                               'pad', self.pad, 'stride', self.stride, ...
                               'method', 'avg', self.opts{:});
      derInputs{2} = [];
      derParams = {} ;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.poolSize ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
        assert(all(obj.pad==0) && all(obj.poolSize==2));
        outputSizes{1} = inputSizes{1};
        outputSizes{1}(1:2) = outputSizes{1}(1:2) .* obj.stride(1:2);
    end

    function obj = UnPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
