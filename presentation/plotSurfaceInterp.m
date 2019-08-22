%create design matrix and output targets for training a NN on surface
%morphology and save them in data file as nnDesignMatrix and nnOutputs
clear
nnCalibrationChannel = 6; %1 indexed
[file, path] = uigetfile('*.mat','Select .mat data file');
if (file == 0)
    return; %canceled
end

dataFile = matfile(strcat(path,file),'Writable',true);
summaryMD = dataFile.summaryMD;
interpPoints = dataFile.surfInterpPoints;
positions = dataFile.stitchedXYZPositions;

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



%calculate DT
tris = num2cell(delaunay(interpPointsImageCoords_um(:,1),interpPointsImageCoords_um(:,2)),2);
%sets of vertices for all DTs
vertices =cellfun(@(vertexIndices) interpPointsImageCoords_um(vertexIndices,:), tris,'UniformOutput',0);

%plot the results of one search together with interpolation
%use maitain calibration data for this
% points = interpPointsImageCoords_um;

% Other LN-the roudned one 
points = load('LNPoints.txt');
%axes start at 0
points = points - repmat(min(points),size(points,1),1);





%interpolate surface
interpN = 5000;
[xSurf, ySurf] = meshgrid(linspace(min(points(:,1)),max(points(:,1)),interpN),...
    linspace(min(points(:,2)),max(points(:,2)),interpN));
% [xSurf, ySurf] = meshgrid(linspace(bounds(1,1),bounds(1,2),interpN),...
%     linspace(bounds(2,1),bounds(2,2),interpN));

zVals = dtInterp(delaunay(points(:,1),points(:,2)), points(:,1),points(:,2),points(:,3),xSurf,ySurf);


figure(1)
h = surf(xSurf,ySurf,-zVals);

colormap viridis
set(h, 'edgecolor','none')
axis equal
view(90,25)

%sweet lighitng
lightangle(-60, 40)
h.FaceLighting = 'gouraud';
h.AmbientStrength = 0.15;
h.DiffuseStrength = 1;
h.SpecularStrength = 0.0;
h.SpecularExponent = 6;
alpha(1)


ax = gca;
ax.XTick = [];
ax.YTick = [];
ax.ZTick = [];
axis off


view(-30,25)

print('Pop LN magellan surface winter','-dtiff')



%% plot lines going to surface
alpha(0.55)

SEARCH_START_DIST = 400.0;
SEARCH_TOLERANCE = 0.1;
N_SAMPLES_THETA = 12;
N_SAMPLES_PHI = 6;

thetas = linspace(0,2*pi, N_SAMPLES_THETA);
%52 degrees is maximum angle of NA
phis = linspace(0, 50.0 / 360.0 * pi * 2.0,N_SAMPLES_PHI);
%convert from XY stage coordinates


[phiGrid, thetaGrid] = meshgrid(phis,thetas);
directionVecs = arrayfun(@(theta,phi) -[cos(theta).*sin(phi), sin(theta).*sin(phi), cos(phi)],...
    thetaGrid, phiGrid,'UniformOutput',0);


hold on
initialPoint = [500, 1000, 300];
initialPoint = [initialPoint(1:2) 300];
scatter3(initialPoint(1),initialPoint(2),-initialPoint(3), 800, 'filled');
distVals = [];
for thetaIndex = 1:N_SAMPLES_THETA
    directionUnitVec = directionVecs{thetaIndex,5};
    dist = computeDistToInterp(vertices, initialPoint, directionUnitVec, initialPoint);
    endPoint = initialPoint + dist*directionUnitVec;
    plot3([initialPoint(1) endPoint(1)],[initialPoint(2) endPoint(2)],-[initialPoint(3) endPoint(3)],'k-','LineWidth',4)
    distVals = [distVals dist];
end
hold off

figure(2)
binedges = linspace(0, 1, 13) .^ 1.5 * 350;
histogram(distVals,binedges);



