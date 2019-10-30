function [] = saveCellTypeParams(cellTypeName, fieldName, fieldVal)
%SAVECELLTYPEPARAMS Summary of this function goes here
%   Detailed explanation goes here
saveFile = matfile(sprintf('CellTypeParams_%s',cellTypeName),'Writable',true);

eval(sprintf('saveFile.%s = fieldVal', fieldName));

end

