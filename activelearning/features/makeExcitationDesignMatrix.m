function [ designMatrix ] = makeExcitationDesignMatrix( queryPoints, interpPoints, summaryMD, brightness, tilePosition )
%return a deisgn matrix with:
%8 entries for position in FOV
%1 entry for vertical distance
%2 entries for position within tile
%1 entry for brightness

SEARCH_START_DIST = 400.0;
SEARCH_TOLERANCE = 2.0;
N_SAMPLING_ANGLES = 8;
%52 degrees is maximum angle of NA
PHI = 40.0 / 360.0 * pi * 2.0;
%convert from XY stage coordinates
[interpPointsImageCoords_um, stagePositionImageCoords_um] = stageToPixelCoordinates();
%calculate DT
tris = num2cell(delaunay(interpPointsImageCoords_um(:,1),interpPointsImageCoords_um(:,2)),2);
%sets of vertices for all DTs
vertices =cellfun(@(vertexIndices) interpPointsImageCoords_um(vertexIndices,:), tris,'UniformOutput',0);
dTheta = pi * 2.0 / N_SAMPLING_ANGLES;

%add distances to interpolation into design matrix
designMatrix = zeros(size(queryPoints,1),N_SAMPLING_ANGLES + 1);
for pointIndex = 1: size(queryPoints,1)
    fprintf('pointIndex %i of %i\n',pointIndex,size(queryPoints,1));
    initialPoint = queryPoints(pointIndex,:);
    %replace x and y coordinates wth the center of the appropriate stage
    %posiiton
    distSq = sum((stagePositionImageCoords_um - repmat(initialPoint(1:2),size(stagePositionImageCoords_um,1),1)).^2,2);
    [~, argmin] = min(distSq);
    initialPoint(1:2) = stagePositionImageCoords_um(argmin,:);
    
    directionVecs = cellfun(@(theta) -[cos(theta).*sin(PHI), sin(theta).*sin(PHI), cos(PHI)],...
        num2cell((0:N_SAMPLING_ANGLES-1)*dTheta),'UniformOutput',0);        
    for  angleIndex = 1:length(directionVecs)
        directionUnitVec = directionVecs{angleIndex};
        initialDist = SEARCH_START_DIST;
        %start with a point outside and then binary line search for the distance
        while (isWithinSurace(initialPoint + directionUnitVec*initialDist) )
            initialDist = initialDist*2;
        end
        designMatrix(pointIndex,angleIndex) = binarySearch(initialPoint, directionUnitVec, 0, initialDist);
    end
    %last column, vertical distance below
    %this one, unlike the others, is specific to the exact point rather
    %than the stage position
    interpVal = getInterpolatedZVal(queryPoints(pointIndex,:));
    if isempty(interpVal)
       designMatrix(pointIndex,end) = 0;
    else        
        designMatrix(pointIndex,end) = queryPoints(pointIndex,3) - interpVal;
    end
end

%add tile position, brightness
designMatrix = (designMatrix - 150) / 100;
%scale tile positions between 0 and 1
tilePosition = tilePosition / (summaryMD.Width * summaryMD.PixelSize_um);
brightness = (brightness - mean(brightness)) ./ std(brightness);
designMatrix = [designMatrix tilePosition brightness];

    function [distance] = binarySearch(initialPoint, direction, minDistance, maxDistance)
%         fprintf('min: %d\tmax: %d\n',minDistance,maxDistance);
        halfDistance = (minDistance + maxDistance) / 2.0;
        %if the distance has been narrowed to a sufficiently small interval, return
        if (maxDistance - minDistance < SEARCH_TOLERANCE)
            distance = halfDistance;
            return
        end
        %check if point is above surface in
        searchPoint = initialPoint + direction*halfDistance;
%         fprintf('search distance: %.0f\n',halfDistance);
        withinSurface = isWithinSurace(searchPoint);
        if (~withinSurface)
            distance = binarySearch(initialPoint, direction, minDistance, halfDistance);
        else
            distance = binarySearch(initialPoint, direction, halfDistance, maxDistance);
        end
    end

    function [zVal] = getInterpolatedZVal(point)
        zVal = [];
        inTriangles = cellfun(@(vertexSet) inpolygon(point(1),point(2),vertexSet(:,1),vertexSet(:,2)),vertices);
        triangleIndex = find(inTriangles);
        if (isempty(triangleIndex))
            return; %outside convex hull
        end
        %else calculate z values to if above or below
        vertexSet = vertices{triangleIndex};
        edge1 =  vertexSet(2,:) - vertexSet(1,:);
        edge2 =  vertexSet(3,:) - vertexSet(1,:);
        normal = cross(edge1,edge2);
        normal = normal / norm(normal);
        %set values for relevant cell surfaces
        %n dot (x - x0) = 0, solve for z coordinate
        zVal =  (point(1:2) - vertexSet(1,1:2) )*normal(1:2)'./ -normal(3) + vertexSet(1,3);
    end

    function [within] = isWithinSurace(point)
        within = 0;
        %figure out which triangle the point is in
        zVal = getInterpolatedZVal(point);
        if isempty(zVal)
           return; %outside convex hull; 
        end
        within = zVal < point(3);
   end

    function [interpPointsImageCoords_um, stagePositionImageCoords_um] = stageToPixelCoordinates()
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
        interpPointsImageCoords_um = [interpPointsPixelCoords * summaryMD.PixelSize_um interpPoints(:,3)];
        stagePositionPixelCoordinates = stagePixelPositions- repmat(pixelOrigin,size(stagePixelPositions,1),1);
        stagePositionImageCoords_um = stagePositionPixelCoordinates * summaryMD.PixelSize_um;
    end

end



