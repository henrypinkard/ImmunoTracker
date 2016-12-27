function [ predLabel, predValue ] = classify( classifier, features, mask, coiIndices, ncoiIndices )
%CLASSIFY use previsouly trained classifier to make predictions of the
%instances specified by mask
%classfiier -- the classfier object created by trainClassfier.m
%features -- n x p data matrix
%mask --logical mask specifying which of the n instances to classfy
%predLabel 0 or 1 depending on the class
%predValue value between 0 and 1 representing certainty of the prediction

%classify
predValue = classifier( features(mask,:)' )';
%override any classifications with manual labels if available
maskIndices = find(mask);
predValue(intersect(maskIndices,coiIndices)) = 1;
predValue(intersect(maskIndices,ncoiIndices)) = 0;
% predValue(
predLabel = predValue > 0.5;

end

