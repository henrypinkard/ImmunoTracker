clear

refCellIndices = [           18516
       46908
       46913
       49090
       49094];

load('DCFeaturesAndLabelsUnstandardized.mat');
% spectra = data.features(


spectra = features(:,114:119);

cellMask = ismember(imarisIndices, refCellIndices);
refVec = mean(spectra(cellMask,:));
refVec = refVec ./ norm(refVec)