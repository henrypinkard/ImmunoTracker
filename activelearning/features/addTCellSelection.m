function [  ] = addTCellSelection( selectionII )
    load('/Users/henrypinkard/Google Drive/Code/MATLAB/CS289 project/data/PreprocessedCMTMRData.mat');
    index = find(imarisIndices == selectionII);
    index0 = index - 1;
    labelledNotTCell(labelledNotTCell == index0) = [];
    labelledTCell = [labelledTCell; index0];
    save('/Users/henrypinkard/Google Drive/Code/MATLAB/CS289 project/data/PreprocessedCMTMRData.mat',...
    'featureNames','features','imarisIndices','labelledTCell','labelledNotTCell','unlabelled');
end

