function dag_viz(net, outputPath)
fid = fopen(outputPath,'w');
fprintf(fid,'digraph dnn {\n');

for iter = 1:numel(net.layers)
    layer = net.layers(iter);
    fprintf(fid,'node%s [label=%s,shape=box];\n',layer.name, ...
            layer.name);

    for jter = 1:numel(layer.inputs)
        fprintf(fid,'var%s -> node%s;\n', ...
                layer.inputs{jter},layer.name);
    end
    for jter = 1:numel(layer.outputs)
        fprintf(fid,'node%s -> var%s;\n', ...
                layer.name,layer.outputs{jter});
    end
end

for iter = 1:numel(net.vars)
    var = net.vars(iter);
    fprintf(fid,'var%s [label=%s];\n',var.name,var.name);
end

fprintf(fid,'}\n');
fclose(fid);
