function [ filteredImg ] = reverseRankFilter( inputImg )
filteredImg = zeros(size(inputImg));
for i = 1:size(inputImg,3)
    for j = 1:size(inputImg,4)
        filteredImg(:,:,i,j) = colfilt(inputImg(:,:,i,j),[3 3],'sliding',@(x) prctile(x,10));
    end
end
end

