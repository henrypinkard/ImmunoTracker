function [ saveFile ] = create_surface_save_file( saveName, surfName, imaris )
saveFile = matfile(saveName);

%initialize fields
saveFile.vertices = single([]);
saveFile.triangles = int32([]);
saveFile.normals = single([]);
saveFile.timeIndex = int32([]);
saveFile.numTriangles = int32([]);
saveFile.numVertices = int32([]);
saveFile.name = surfName;

if (nargin == 3) %saving from stitched file, so get time info
    dataset = imaris.GetDataSet;
    numTP = dataset.GetSizeT;
    timeCalibration = cell(numTP,1);
    for i = 0:numTP-1
       timeCalibration{1+i} = dataset.GetTimePoint(i); 
    end
    saveFile.timeCalibration = timeCalibration;
end


end

