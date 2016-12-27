function [vertDistBelowSurface, normalAngleWithVertical, normalProjection] = calcInterpFeatures(...
    features, featureNames,summaryMD, interpPoints)

pixelSize = summaryMD.PixelSize_um;

stitchedPosX = features(:,strcmp(featureNames,'Stitched Position X'));
stitchedPosY = features(:,strcmp(featureNames,'Stitched Position Y'));
stitchedPosZ = features(:,strcmp(featureNames,'Stitched Position Z'));

indexX = floor(stitchedPosX ./ pixelSize) + 1;
indexY = floor(stitchedPosY ./ pixelSize) + 1;
tilePosX = features(:,strcmp(featureNames,'Position X'));
tilePosY = features(:,strcmp(featureNames,'Position Y'));
tileIndexX = floor(tilePosX ./ pixelSize) + 1;
tileIndexY = floor(tilePosY ./ pixelSize) + 1;

cellSurfaceXYPixelIndices = [indexX indexY];
cellSurfaceXYTilePixelIndices = [tileIndexX tileIndexY];


%use stage position data to align pixel coordiantes of image and
%interpolation coordinates of stage

posList = summaryMD.InitialPositionList;
stagePositions = cell2mat(cellfun(@(entry) cell2mat(entry.DeviceCoordinatesUm.XYStage)',...
    posList,'UniformOutput', false))';

%figure out pixel size of stitched image
rowColIndices = cell2mat(cellfun(@(entry) [entry.GridColumnIndex entry.GridRowIndex],...
    posList,'UniformOutput',false)');
gridDims = range(rowColIndices) + 1;
tileDim = [summaryMD.Width - summaryMD.GridPixelOverlapX summaryMD.Height - summaryMD.GridPixelOverlapY];
stitchedImageSize = [tileDim .* gridDims] + [summaryMD.GridPixelOverlapX summaryMD.GridPixelOverlapY];

%convert interpolation points to pixel coordinates
linearTransform = reshape(str2double(strsplit(summaryMD.AffineTransform,'_')),2,2)';
stagePixelPositions = (stagePositions / linearTransform );
centerPixel = min(stagePixelPositions) + 0.5*range(stagePixelPositions);
pixelOrigin = centerPixel - 0.5*stitchedImageSize;
interpPointsPixelCoords = interpPoints(:,1:2) / linearTransform - repmat(pixelOrigin,size(interpPoints,1),1);
stagePositionPixelCoordinates = stagePixelPositions- repmat(pixelOrigin,size(stagePixelPositions,1),1);
[zVals, ~] = calcInterpedValues(interpPointsPixelCoords, interpPoints(:,3), cellSurfaceXYPixelIndices);

%replace all undefined values outside convex hull with the closest value
outOfBoundsIndices = find(zVals == -1);
for i = outOfBoundsIndices'
    distanceToOthers = sqrt(sum((cellSurfaceXYPixelIndices - repmat(cellSurfaceXYPixelIndices(i,:),size(cellSurfaceXYPixelIndices,1),1) ).^2,2));
    %look only at ones with defined values
    closestIndex = find(distanceToOthers == min(distanceToOthers(zVals ~= -1)),1 );
    zVals(i) = zVals(closestIndex);
end

%find normals at centers of stage positions
[~, stagePosNormals] = calcInterpedValues(interpPointsPixelCoords, interpPoints(:,3), stagePositionPixelCoordinates);
undefinedNormalMask = cell2mat(cellfun(@(a) length(a) == 1 && a == -1,stagePosNormals,'UniformOutput',false));
stagePosNormals = cell2mat(stagePosNormals(~undefinedNormalMask));
%normalize
stagePosNormals = stagePosNormals ./ repmat(sqrt(sum(stagePosNormals.^2,2)),1,3);
validStagePositions = stagePositionPixelCoordinates(~undefinedNormalMask,:);
%ditance between stage positions (in pixel coords)
interPosDistancePix = sqrt(sum((validStagePositions(1,:) - validStagePositions(2,:)).^2));
weightedAvgDistance = interPosDistancePix*sqrt(2);
%take weighted average of up to 4 positions
normals = zeros(size(cellSurfaceXYPixelIndices,1),3);
for i = 1:size(cellSurfaceXYPixelIndices,1)
    distanceToSP = sqrt(sum((validStagePositions - repmat(cellSurfaceXYPixelIndices(i,:),size(validStagePositions,1),1) ).^2,2));
    withinDistanceIndices = find(distanceToSP <= weightedAvgDistance);
    %take weighted average of all normals in range
    normalsInRange = stagePosNormals(withinDistanceIndices,:);
    weights = weightedAvgDistance - distanceToSP(withinDistanceIndices);
    weights = weights ./ sum(weights);
    normals(i,:) = sum(normalsInRange .* repmat(weights,1,3),1);
    %renormalize
    normals(i,:) = normals(i,:) ./ norm( normals(i,:) );
end

vertDistBelowSurface =  stitchedPosZ - zVals;
normalAngleWithVertical = acos(normals(:,3));

% take xy component of normal
normalXYProj = normals(:,1:2);
doublePosMask = normalXYProj(:,1) >= 0 & normalXYProj(:,2) >= 0;
doubleNegMask = normalXYProj(:,1) < 0 & normalXYProj(:,2) < 0;
xNegMask = normalXYProj(:,1) < 0 & normalXYProj(:,2) >= 0;
yNegMask = normalXYProj(:,1) >= 0 & normalXYProj(:,2) < 0;

%project each surfaces coordinates within tile onto its normal vector
cellCoords = cellSurfaceXYTilePixelIndices;
%modify so that all projections > 0
cellCoords(doubleNegMask,:) =  cellCoords(doubleNegMask,:) - 410;
cellCoords(xNegMask,:) = [(cellCoords(xNegMask,1)- 410), cellCoords(xNegMask,2)];
cellCoords(yNegMask,:) = [cellCoords(yNegMask,1), (cellCoords(yNegMask,2) - 410)];
normalProjection = dot(cellCoords',normalXYProj')' ./ sqrt(sum(normalXYProj.^2,2));

end


function [zVals, normals] = calcInterpedValues(interpolationPixelCoordsXY, zCoordinates, queryPoints)
pixelSize = 0.351;
%calculate DT
tris = delaunay(interpolationPixelCoordsXY(:,1),interpolationPixelCoordsXY(:,2));
zVals = -1*ones(size(queryPoints,1),1);
normals = cell(size(queryPoints,1),1);
for i = 1:length(tris)
    %get vertices of this triangle
    vertexIndices = tris(i,:);
    verticesX =  interpolationPixelCoordsXY(vertexIndices,1);
    verticesY =  interpolationPixelCoordsXY(vertexIndices,2);
    %get indices of surface within
    inTriangleMask = inpolygon(queryPoints(:,1),queryPoints(:,2),verticesX,verticesY);
    %calculate normal
    %multiply by pixel size for x and y so all units are um
    verticesX_um = pixelSize*verticesX;
    verticesY_um = pixelSize*verticesY;
    verticesZ_um = zCoordinates(vertexIndices);
    allVertices = [verticesX_um verticesY_um verticesZ_um];
    edge1 =  allVertices(2,:) - allVertices(1,:);
    edge2 =  allVertices(3,:) - allVertices(1,:);
    normal = cross(edge1,edge2);
    normal = normal / norm(normal);
    normals(inTriangleMask) = {normal};
    %set values for relevant cell surfaces
    %n dot (x - x0) = 0, solve for z coordinate
    interpZ = @(xy) (xy - repmat(allVertices(1,1:2),size(xy, 1),1))*normal(1:2)'./ -normal(3) + allVertices(1,3);   
    %convert query points to pixel coords
    zVals(inTriangleMask) = interpZ(queryPoints(inTriangleMask,:).*pixelSize );
end

%replace empty entries (outside interp area)
normals(cellfun(@isempty,normals)) = {-1};
end
