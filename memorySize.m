function sz = memorySize(var)
    detail = whos('var');
    sz = detail.bytes / 2^20;
