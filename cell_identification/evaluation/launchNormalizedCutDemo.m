clear

%Imaris index 35908 -- in farrred duct
% Imaris index 20305 -- touching green and farred dendrite
%Imaris index 156469 -- green red lots of overlap
%Imaris index 169026 -- green red overlap
%Imaris index 168870 -- T cell overlap with green-red DC

% Green Red good example 156381
%same one differ TP 156288 
%another one 169026

referenceVector = [0.1024    0.0700    0.0953    0.0691    0.9660    0.1939]; %CMTMR
channelOffsets = [8 12 16 12 10 8];
file = matfile('CMTMRTCellAndNonTCellMasksAndImageData.mat');
imarisIndices = file.imarisIndices;
ii = 156288;
i = find(imarisIndices{1} == ii);
img = file.imageData(i,1);
mask = file.masks(i,1);
% xtTransferImageData(img{1});
clusterAndCalcFeatures(img{1},mask{1},channelOffsets,referenceVector);