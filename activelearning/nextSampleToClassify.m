function [ index ] = nextSampleToClassify( features, atCurrentTPMask, coiIndices, ncoiIndices )
%NEXTSAMPLETOCLASSIFY 
%Use the active learning query strategy to determine which unlabelled
%instance should be classified next, and return its index within the
%nxp data matrix features


%for now do a dumb thing and just return an unlabelled instance at the
%current timepoint
unlabelledAtCurrentTP = atCurrentTPMask;
unlabelledAtCurrentTP(coiIndices) = 0;
unlabelledAtCurrentTP(ncoiIndices) = 0;

unlabelledAtCurrentTPIndices = find(unlabelledAtCurrentTP);

%index should be 1-based
index = unlabelledAtCurrentTPIndices(1);
end

