classdef Pooling < dagnn.Filter
    properties
        method = 'max'
        poolSize = [1 1]
        opts = {'cuDNN'}
    end

    methods
        function outputs = forward(self, inputs, params)
            if numel(inputs) >= 2
                % compute the argmax
                sz = size(inputs{1});
                onemap = ones([floor(sz(1:2)/2) sz(3:4)],'single');
                if isa(inputs{1}, 'gpuArray') || isa(inputs{2}, 'gpuArray')
                    onemap = gpuArray(onemap);
                end
                argmax = vl_nnpool(inputs{2}, self.poolSize, onemap, ...
                                   'pad', self.pad, 'stride', self.stride, ...
                                   'method', self.method, self.opts{:});
                % apply the argmax
                outputs{1} = vl_nnpool(inputs{1}.*argmax, self.poolSize, ...
                                       'pad', self.pad, 'stride', self.stride,...
                                       'method', self.method, self.opts{:});
            else
                outputs{1} = vl_nnpool(inputs{1}, self.poolSize, ...
                                       'pad', self.pad, ...
                                       'stride', self.stride, ...
                                       'method', self.method, ...
                                       self.opts{:}) ;
            end
        end

        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
            if numel(inputs) >= 2
                derInputs{1} = vl_nnpool(inputs{2},self.poolSize,derOutputs{1},...
                                         'pad',self.pad,'stride',self.stride,...
                                         'method',self.method,self.opts{:});
                derInputs{2} = [];
            else
                derInputs{1} = vl_nnpool(inputs{1}, self.poolSize, derOutputs{1}, ...
                                         'pad', self.pad, ...
                                         'stride', self.stride, ...
                                         'method', self.method, ...
                                         self.opts{:}) ;
            end
            derParams = {} ;
        end

        function kernelSize = getKernelSize(obj)
            kernelSize = obj.poolSize ;
        end

        function outputSizes = getOutputSizes(obj, inputSizes)
            outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
            outputSizes{1}(3) = inputSizes{1}(3) ;
        end

        function obj = Pooling(varargin)
            obj.load(varargin) ;
        end
    end
end
