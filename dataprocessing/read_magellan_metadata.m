function [posList, imageWidth, imageHeight, xPixelOverlap, yPixelOverlap, pixelSize, numTimePoints] = read_magellan_metadata( directory )
%READ_MAGELLAN_METADATA
%   read calibrtion and posiotion metadata from a Micro-Magellan dataset

% Read micromanager file and get needed data
javaaddpath('C:\Users\hpinkard\Dropbox\Henry\Code\Builds\Magellan.jar');
import acq.*;
import misc.*;
% Read micromanager dataset to get metadata and position info;
mmData = MultiResMultipageTiffStorage(directory);
summaryMD = mmData.getSummaryMetadata;
pixelSize = MD.getPixelSizeUm(summaryMD);
imageHeight = MD.getHeight(summaryMD);
imageWidth = MD.getWidth(summaryMD);
xPixelOverlap = MD.getPixelOverlapX(summaryMD);
yPixelOverlap = MD.getPixelOverlapY(summaryMD);
posList = MD.getInitialPositionList(summaryMD);
numTimePoints = mmData.getNumFrames();
mmData.close;
clear mmData;
end

