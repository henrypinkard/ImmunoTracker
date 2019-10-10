function [refVector] = getReferenceVector(cellTypeName)
%GETCELLTYPEPARAMS Returns needed for surface creation.
%StoreSurfaceCreationParams.m needs to have been run before calling this
saveFile = matfile(sprintf('CellTypeParams_%s',cellTypeName),'Writable',false);

refVector = saveFile.referenceVector;

end
