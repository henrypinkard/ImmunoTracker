%% Preprocess data
%read in data and output nxp data matrix + 0-indexed labels
clear
load('rawData.mat')
%T cells Indices are Imaris indices--same as one s in Stats(1).IDs
tCellMask = ismember(statistics(1).Ids,TCellIndices);
%%%For resaving manuaaly
%save('rawData.mat','TCellIndices','statistics')
featureNames = {statistics.Name}';
statistics(96).Values(TCellIndices);
% read Imaris indices
imarisIndices = statistics(96).Ids;
%TP 14 and onward are unlabeld
unlabelledMask = statistics(96).Values >= 14;
%remove the one labeld t cell at time 14
tCellMask(unlabelledMask) = 0;
notTCellMask = ~tCellMask & statistics(96).Values < 14;

%make data matrix
rawFeatures = cell2mat({statistics.Values});

%remove data points within _ um of edges of tiles
distanceFromEdge = 3;
distanceToBorder = min([rawFeatures(:,88), rawFeatures(:,89),...
    max(rawFeatures(:,88)) - rawFeatures(:,88),  max(rawFeatures(:,89)) - rawFeatures(:,89) ],[],2);
inCenter = rawFeatures(:,88) > distanceFromEdge & rawFeatures(:,88) < (max(rawFeatures(:,88)) - distanceFromEdge) &...
    rawFeatures(:,89) > distanceFromEdge & rawFeatures(:,89) < (max(rawFeatures(:,89)) - distanceFromEdge);
% sum(inCenter(labelledTCell+1))


%visulaized excluded edges
% plot(rawFeatures(:,88),rawFeatures(:,89),'k.')
% hold on
% plot(rawFeatures(inCenter,88),rawFeatures(inCenter,89),'r.')
% hold off


imarisIndices = imarisIndices(inCenter,:);
features = rawFeatures(inCenter,:);
labelledTCell = find(tCellMask(inCenter)) - 1;
labelledNotTCell = find(notTCellMask(inCenter)) - 1;
unlabelled = find(unlabelledMask(inCenter)) - 1;

% size(TCellIndices)
% size(labelledTCell)


save('PreprocessedTCellData.mat','labelledTCell','labelledNotTCell','unlabelled', 'features','featureNames','imarisIndices')

