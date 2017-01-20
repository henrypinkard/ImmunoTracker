%Get surfaces object selected in surpass and save it, along with any tracks
function [] = save_surfaces(imarisIndex)
batchSize = 500;

%connect to imaris
vImarisLib = ImarisLib;
imaris = vImarisLib.GetApplication(imarisIndex);
if (isempty(imaris))
    msgbox('Wrong imaris index');
    return;
end

% Create matlab file to save surface data in
filename = imaris.GetCurrentFileName;
surface = imaris.GetFactory.ToSurfaces(imaris.GetSurpassSelection);
surfName = surface.GetName;

%generate saving name
imsFileFullPath = strsplit(char(filename),'.');
imsFilePathAndName = imsFileFullPath{1};
%save in same directory as data
saveName = strcat(imsFilePathAndName,'_',char(surfName),'.mat');
%delete exisiting file with same name
if (exist(saveName,'file') == 2)
    if (strcmp(questdlg('Delete exisiting file'),'Yes'))
        delete(saveName);
    else
        return;
    end
end
saveFile = create_surface_save_file(saveName,surfName,imaris);

% write batches of surfaces at a time so as not to overflow memory
for i = 0:ceil((surface.GetNumberOfSurfaces)/batchSize ) - 1
    startIndex = batchSize*i;
    endIndex = min(surface.GetNumberOfSurfaces-1, batchSize*(i+1) - 1);
    fprintf('%d to %d of %d\n',startIndex+1, endIndex +1, surface.GetNumberOfSurfaces)
    surfList = surface.GetSurfacesList(startIndex:endIndex);
    
    saveFile.vertices(size(saveFile, 'vertices', 1)+1:size(saveFile, 'vertices', 1)+size(surfList.mVertices,1),1:3) = surfList.mVertices;
    saveFile.triangles(size(saveFile, 'triangles', 1)+1:size(saveFile, 'triangles', 1)+size(surfList.mTriangles,1),1:3) = surfList.mTriangles;
    saveFile.normals(size(saveFile, 'normals', 1)+1:size(saveFile, 'normals', 1)+size(surfList.mNormals,1),1:3) =  surfList.mNormals;
    saveFile.timeIndex(size(saveFile, 'timeIndex',1)+1 : size(saveFile, 'timeIndex',1)+length(surfList.mTimeIndexPerSurface),1) = surfList.mTimeIndexPerSurface;
    saveFile.numTriangles(size(saveFile, 'numTriangles',1)+1 : size(saveFile, 'numTriangles',1)+length(surfList.mNumberOfTrianglesPerSurface),1) = surfList.mNumberOfTrianglesPerSurface;
    saveFile.numVertices(size(saveFile, 'numVertices',1)+1 : size(saveFile, 'numVertices',1)+length(surfList.mNumberOfVerticesPerSurface),1) =  surfList.mNumberOfVerticesPerSurface;
end

%save tracks
saveFile.trackEdges = surface.GetTrackEdges;


% Copy to backup directory on different drive
% if (any(imsFilePathAndName == '\'))
%     dirs = strsplit(imsFilePathAndName,'\');
% else
%     dirs = strsplit(imsFilePathAndName,'/');
% end
% copyfile(saveName, strcat('D:\Data\Henry\Surface autosave backups\',dirs{end-1},'_',dirs{end},'_',char(surfName),'.mat') );

end

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
