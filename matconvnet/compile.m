addpath matlab;

% compile vl_nnrelu.cu individually (mexcuda only for R15b)
%mexcuda COPTIMFLAGS='-O3' -largeArrayDims -output matlab/mex/vl_nnrelu matlab/src/vl_nnrelu.cu -lstdc++ -lc

% compile with built-in function
vl_compilenn('enableImreadJpeg', true, 'enableGpu', true, 'cudaRoot', '/usr/local/cuda',...
             'cudaMethod', 'nvcc', 'enableCudnn', true, 'cudnnRoot', 'local/cudnn');
