function [ index ] = nextSampleToClassify( currentTPPredValue, mask )
%NEXTSAMPLETOCLASSIFY 
%Use the active learning query strategy to determine which unlabelled
%instance at the current time point should be classified next, and return
%its index within the nxp design matrix. 
%currentTPPredValue -- value between 0 and 1 that represents confidence of
%prediction
%mask -- binary mask for which instances in consideration (unlabelled and
%displayed at current TP)

designMatrixIndices = find(mask);

certainty = abs(currentTPPredValue - 0.5);
[~,sortedIndices] = sort(certainty);

%index should be 1-based
index = designMatrixIndices(sortedIndices(1));

histogram(currentTPPredValue,100);
set(gca,'YScale','log')
end


