function [xSurfaces] = func_addsurfacestosurpass(xImarisApp, file, batchSize, xSurfaces, surfIndicesToAdd)
%surfIndicesToAdd is Imaris indices, so 0 indexed

if (nargin == 3)
    addAll = true;
else
    addAll = false;
end


if (addAll)
    xSurfaces = xImarisApp.GetFactory.CreateSurfaces;
    xSurfaces.SetName(char(file.name));
    xImarisApp.GetSurpassScene.AddChild(xSurfaces,-1);
    numBatches = ceil( length(file.timeIndex) / batchSize); 
    trianglesAdded = 0;
    verticesAdded = 0;
else
    numBatches = ceil(length(surfIndicesToAdd) / batchSize); 
    numVertices = file.numVertices;
    numTriangles = file.numTriangles;
    getOffsetIndex = @(surfIndex, countVec) sum(countVec(1:surfIndex-1));
end

for i = 0:numBatches - 1
    fprintf('batch %d of %d\n',i+1,numBatches)
    %add one batch of surfaces
    if (addAll)
        surfaceIndexStart = i*batchSize + 1;
        surfaceIndexEnd = min((i+1)*batchSize, size(file,'timeIndex',1));
        try
            nVert = file.numVertices(surfaceIndexStart:surfaceIndexEnd,:);
            nTri = file.numTriangles(surfaceIndexStart:surfaceIndexEnd,:);
            verticesInBatch = sum(nVert);
            trianglesInBatch = sum(nTri);
            timeInd = file.timeIndex(surfaceIndexStart:surfaceIndexEnd,:);
            norms = file.normals(verticesAdded + 1: verticesAdded + verticesInBatch,:);
            verts = file.vertices(verticesAdded + 1: verticesAdded + verticesInBatch,:);
            tris = file.triangles(trianglesAdded + 1: trianglesAdded + trianglesInBatch,:);
        catch            
            %sometimes files get corrupted, so let user know
            fprintf('Load failure');
            return;
        end   
        trianglesAdded = trianglesAdded + trianglesInBatch;
        verticesAdded = verticesAdded + verticesInBatch;  
    else
        if i == numBatches - 1
            surfacesInBatch = length(surfIndicesToAdd) - (numBatches-1) * batchSize;
        else
            surfacesInBatch = batchSize;
        end
        nTri = zeros(surfacesInBatch,1);
        nVert = zeros(surfacesInBatch,1);
        timeInd = zeros(surfacesInBatch,1);
        verts = cell(surfacesInBatch,1);
        tris = cell(surfacesInBatch,1);
        norms = cell(surfacesInBatch,1);
        
        %pick out surfaces individually, add them to the batch, then add
        %the batch to imaris
        for j = 1:surfacesInBatch
            surfIndex = surfIndicesToAdd(i * batchSize + j);
            
            timeInd(j) = file.timeIndex(surfIndex + 1,:);
            nTri(j) = numTriangles(surfIndex+1);
            nVert(j) = numVertices(surfIndex+1);
            vertOffset = getOffsetIndex(surfIndex + 1, numVertices);
            triOffset = getOffsetIndex(surfIndex + 1, numTriangles);
            norms{j} = file.normals( vertOffset + 1: vertOffset + numVertices(surfIndex+1) ,:);
            verts{j} = file.vertices( vertOffset + 1: vertOffset + numVertices(surfIndex+1) ,:);
            tris{j} = file.triangles( triOffset + 1: triOffset + numTriangles(surfIndex+1) ,:);
        end
        
        verts = cell2mat(verts);
        tris = cell2mat(tris);
        norms = cell2mat(norms);
    end
    
    xSurfaces.AddSurfacesList(verts,nVert,tris,nTri,norms,timeInd);
end

%check if file has track edges
if (any(strcmp('trackEdges',who(file))))
    xSurfaces.SetTrackEdges(file.trackEdges);
end

end
