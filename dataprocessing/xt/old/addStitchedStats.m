%% This script adds stats for position and time index in larger stitched image
%TODO: this can be deleted once stitch and save is verified to work


%get file corresponding to surfaces
[filename, pathname] = uigetfile('*.mat');
if (filename == 0)
    return;
end
surfFile = matfile(strcat(pathname, filename),'Writable',true);


%MM data for reading metadata and positon info
%get file
[filename, pathname] = uigetfile();
if (filename == 0)
    return;
end
mmDataDir = pathname;


% Read micromanager file and get needed data
javaaddpath('C:\Users\hpinkard\Dropbox\Henry\Code\Matlab\Imaris\MMJ_.jar');
javaaddpath('C:\Users\hpinkard\Dropbox\Henry\Code\Matlab\Imaris\MMCoreJ.jar');
import org.micromanager.acquisition.*
import org.micromanager.utils.*
import mmcorej.*
% Read micromanager dataset to get metadata and position info;
mmData = TaggedImageStorageMultipageTiff(mmDataDir, 0, [],0,1,1);
summaryMD = mmData.getSummaryMetadata;
pixelSize = MDUtils.getPixelSizeUm(summaryMD);
imageHeight = MDUtils.getHeight(summaryMD);
imageWidth = MDUtils.getWidth(summaryMD);
xPixelOverlap = summaryMD.getInt('GridPixelOverlapX');
yPixelOverlap = summaryMD.getInt('GridPixelOverlapY');
posList = summaryMD.getJSONArray('InitialPositionList');
numTimePoints = mmData.lastAcquiredFrame + 1;
%figure out if acq was aborted mid time point
if (isempty(mmData.getImage(0,0,numTimePoints - 1, posList.length -1)))
    numTimePoints = numTimePoints - 1;
end
mmData.close;
clear mmData;


%get stat struct
stats = surfFile.stats;

pxIdx = find(ismember({stats.Name},'Tile Position X'));
pyIdx = find(ismember({stats.Name},'Tile Position Y'));
pzIdx = find(ismember({stats.Name},'Tile Position Z'));
tIdxIdx = find(ismember({stats.Name},'Time Index'));

%rename original position stats
stats(pxIdx).Name = 'Tile Position X';
stats(pyIdx).Name = 'Tile Position Y';
stats(pzIdx).Name = 'Tile Position Z';

timeIndex = stats(tIdxIdx).Values;
%time index stat is one based
stitchedTimeIndex = mod(timeIndex - 1,numTimePoints);
%add actual time index in
stats(tIdxIdx).Values = stitchedTimeIndex;

posIndices = floor(double(timeIndex) ./ numTimePoints);
posIndicesCell = num2cell(posIndices);

rows = cellfun(@(index) posList.get(index).getInt('GridRowIndex'),posIndicesCell);
cols = cellfun(@(index) posList.get(index).getInt('GridColumnIndex'),posIndicesCell);
%calculate offset to translate from individual field of view to proper
%position in stitched image
offsets = [(cols * (imageWidth - xPixelOverlap)) * pixelSize, (rows * (imageHeight - yPixelOverlap)) * pixelSize];

singlestruct = @(name, values) struct('Ids',stats(1).Ids,'Name',name,'Values',values,'Units','');

stats(length(stats) + 1) = singlestruct('Stitched Position X',stats(pxIdx).Values + offsets(:,1));
stats(length(stats) + 1) = singlestruct('Stitched Position Y',stats(pyIdx).Values + offsets(:,2));
stats(length(stats) + 1) = singlestruct('Stitched Position Z',stats(pzIdx).Values );

%sort into alphabetical order
[~, statOrder] = sort({stats.Name});
stats = stats(statOrder);
%resave
surfFile.stats = stats;

