function [norms, distancesToMedian, minDistancesToAny] = intensityvectordistancesort(stats,useChannels,imarisObjectIDs, objectIndexMask )
% INTENSITYVECTORDISTANCESORT  Calculate distances from point on unit
% sphere in n-D vector space
% norms = length of n-D vectors
% distancesToMedian = distance to the median value of the vectors of
% selected indices
%
% objectIndexMask is matlab indexed mask 

%TODO: subtract channel offsets from new data
%TODO: maybe this works better with intensity medians?

vIdx = find(ismember({stats.Name},'Intensity Mean - Channel 1'));
bIdx = find(ismember({stats.Name},'Intensity Mean - Channel 2'));
gIdx = find(ismember({stats.Name},'Intensity Mean - Channel 3'));
yIdx = find(ismember({stats.Name},'Intensity Mean - Channel 4'));
rIdx = find(ismember({stats.Name},'Intensity Mean - Channel 5'));
frIdx = find(ismember({stats.Name},'Intensity Mean - Channel 6'));

[norms, normalizedVecs] = createnormalizedvecs(objectIndexMask);

%try finding closest to a known cell
%get matlab indices of only reference surfaces
refIndices = find( ismember(stats(1).Ids,double(imarisObjectIDs)));
[refNorms, refVecs] = createnormalizedvecs(refIndices);
%get normalized intensity vectors of all positions
allPositions = cell2mat(refVecs);
%take median of each component of set of refernce normalized vectors
medianPosition = median(allPositions,1);
%normaize median reference vector
medianPosition = medianPosition ./ norm(medianPosition);


distancesToMedian = cellfun(@(v) norm(v - medianPosition),normalizedVecs);
minDistancesToAny = cellfun(@(v) min(norm(bsxfun(@minus,v,allPositions))),normalizedVecs);


%show histogram
% figure(2)
% hist(distancesToMedian,500);
% ask for user input of cutoff
% cutoff = input('distance cutoff? ')
% [~, i] = sort(distances);
% sortedIds = imarisIndices(i); 

    function [magnitudes, vectors] = createnormalizedvecs(matlabIndexMask)
        %create 6 dimensional vector with intensity stats
        intensityMat = [stats(vIdx).Values(matlabIndexMask), stats(bIdx).Values(matlabIndexMask), stats(gIdx).Values(matlabIndexMask),...
            stats(yIdx).Values(matlabIndexMask), stats(rIdx).Values(matlabIndexMask), stats(frIdx).Values(matlabIndexMask)];
        %Use only channels requested
        intensityMat = intensityMat(:,find(useChannels));
        
        intensityVecs = mat2cell(intensityMat,ones(size(intensityMat,1),1),size(intensityMat,2));
        %normalize to a length of 1
        magnitudes = cellfun(@norm,intensityVecs);
        
        %normalized vectors give coordinates for each element on the surface of an n-dimensional hypersphere
        vectors = mat2cell(bsxfun(@rdivide,intensityMat,magnitudes),ones(size(intensityMat,1),1),size(intensityMat,2));    
    end

end