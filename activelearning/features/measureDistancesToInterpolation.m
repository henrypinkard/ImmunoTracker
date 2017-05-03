function [ distances, distancesStagePositioned ] = measureDistancesToInterpolation( queryPoints, interpPoints, summaryMD)
%return an N x Theta samples x phi samples tensor of distances to
%interpolation. entries for a given point with phi == 0 represent
%vertical distance to surface and will be identical regardless
%of theta
SEARCH_START_DIST = 400.0;
SEARCH_TOLERANCE = 0.1;
N_SAMPLES_THETA = 12;
N_SAMPLES_PHI = 6;

thetas = linspace(0,2*pi, N_SAMPLES_THETA);
%52 degrees is maximum angle of NA
phis = linspace(0, 50.0 / 360.0 * pi * 2.0,N_SAMPLES_PHI);
%convert from XY stage coordinates
[interpPointsImageCoords_um, stagePositionImageCoords_um] = stageToPixelCoordinates();
%calculate DT
tris = num2cell(delaunay(interpPointsImageCoords_um(:,1),interpPointsImageCoords_um(:,2)),2);
%sets of vertices for all DTs
vertices =cellfun(@(vertexIndices) interpPointsImageCoords_um(vertexIndices,:), tris,'UniformOutput',0);

%add distances to interpolation into design matrix
distances = zeros(size(queryPoints,1), N_SAMPLES_THETA, N_SAMPLES_PHI);
distancesStagePositioned = zeros(size(queryPoints,1), N_SAMPLES_THETA, N_SAMPLES_PHI);

[phiGrid, thetaGrid] = meshgrid(phis,thetas);
directionVecs = arrayfun(@(theta,phi) -[cos(theta).*sin(phi), sin(theta).*sin(phi), cos(phi)],...
    thetaGrid, phiGrid,'UniformOutput',0);

%plot the results of one search together with interpolation
%use maitain calibration data for this
% plotInterpolatedSurface(interpPointsImageCoords_um, [400 900; 1150 2000]);
plotInterpolatedSurface(interpPointsImageCoords_um, [100 1400; 650 2400]);


hold on
initialPoint = queryPoints(2810,:);
initialPoint = [initialPoint(1:2) 300];
scatter3(initialPoint(1),initialPoint(2),-initialPoint(3),200, 'filled');
distVals = [];
for thetaIndex = 1:N_SAMPLES_THETA
    directionUnitVec = directionVecs{thetaIndex,5};
    dist = computeDistToInterp(initialPoint, directionUnitVec, initialPoint);
    endPoint = initialPoint + dist*directionUnitVec;
    plot3([initialPoint(1) endPoint(1)],[initialPoint(2) endPoint(2)],-[initialPoint(3) endPoint(3)],'k-','LineWidth',2)
    distVals = [distVals dist];
end
hold off
figure(2)
binedges = linspace(0, 1, 13) .^ 1.5 * 350;
histogram(distVals,binedges);

% print('interpolated LN with search distance','-dtiff')




for pointIndex = 1: size(queryPoints,1)
    fprintf('pointIndex %i of %i\n',pointIndex,size(queryPoints,1));
    initialPoint = queryPoints(pointIndex,:);
    %replace x and y coordinates wth the center of the appropriate stage posiiton
    distSq = sum((stagePositionImageCoords_um - repmat(initialPoint(1:2),size(stagePositionImageCoords_um,1),1)).^2,2);
    [~, argmin] = min(distSq);
    initialPointStagePositioned = queryPoints(pointIndex,:);
    initialPointStagePositioned(1:2) = stagePositionImageCoords_um(argmin,:);
    
    for angleIndex = 0:length(directionVecs(:))-1
        thetaIndex = mod(angleIndex, N_SAMPLES_THETA) + 1;
        phiIndex = floor(angleIndex / N_SAMPLES_THETA) + 1;
        directionUnitVec = directionVecs{thetaIndex,phiIndex};
        if phiIndex == 1
            %phi = 0 so just get interpolation value from overhead
            interpVal = getInterpolatedZVal(queryPoints(pointIndex,:));
            if isempty(interpVal)
                value = 0;
                valueSP = 0;
            else
                value = max(0,queryPoints(pointIndex,3) - interpVal);
                valueSP = max(0,queryPoints(pointIndex,3) - interpVal);
            end
        else
            [value, valueSP] = computeDistToInterp(initialPoint, directionUnitVec, initialPointStagePositioned);
        end
        distances(pointIndex,thetaIndex,phiIndex) = value;
        distancesStagePositioned(pointIndex,thetaIndex,phiIndex) = valueSP;
    end
end


%%%%%functions%%%%%%%

    function [val, valSP] = computeDistToInterp(initialPoint, directionUnitVec,initialPointStagePositioned)
        
        %binary line search to find distance to interpolation
        initialDist = SEARCH_START_DIST;
        %start with a point outside and then binary line search for the distance
        while isWithinSurace(initialPoint + directionUnitVec*initialDist)
            initialDist = initialDist*2;
        end
        val =   binarySearch(initialPoint, directionUnitVec, 0, initialDist);
        %%%
        initialDist = SEARCH_START_DIST;
        %start with a point outside and then binary line search for the distance
        while isWithinSurace(initialPointStagePositioned + directionUnitVec*initialDist)
            initialDist = initialDist*2;
        end
        valSP = binarySearch(initialPointStagePositioned, directionUnitVec, 0, initialDist);
        
    end

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



