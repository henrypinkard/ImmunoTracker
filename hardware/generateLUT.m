%start with elapsed time since start of line
%position in scan pattern is sinusoidal function of time

%start with measured time position
%LUT to get position in space
%index high resolution space position into indices for 8x8 pattern and
%multipliers for combining them

linescan_time_us = 127.2; %TODO change
numGridPoints = 8;


%begin: position along line normalized between 0 and 1;
normalizedTime = 0:0.01:1;

time_to_space = @(elapsed_us)  (1 +cos( elapsed_us ./linescan_time_us  * 2*pi))/2 ;
%calcualte the normalized spatial position for each possible clock value
%along a line
spatialPos = time_to_space(0:ceil(linescan_time_us));

%calculate the bins that correspond to it
gridTileSize = 1 ./ numGridPoints;
%calcualte which two to take a weight average between
boundingIndices = [floor(spatialPos / gridTileSize); floor(spatialPos / gridTileSize) + 1];
%calculate weightings for that weighted average
weights = mod(spatialPos,gridTileSize) ./ gridTileSize; 



plot(boundingIndices(1,:),'o-')
hold on
plot(weights*numGridPoints,'o-')
hold off