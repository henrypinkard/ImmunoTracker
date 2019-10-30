function [ tracksWithVecs ] = getChemokineAndMigrationVecs( tracks )
%GETCHEMOKINEGRADVECTOR
% Find the vector pointing in the direction of expected movement based
% on the positions of all other cells assuming exponential decay of chemokine gradients

for i = 1:length(tracks)
    %for each track and each time point, sum the vectors pointing to all
    %other cells weighted by exponential decay for distance
    for t = [tracks{i}.t] %at each time point
        cellTimeIndex = find([tracks{i}.t] == t);
        %get vector of migration from this time point to next
        if cellTimeIndex ~= length([tracks{i}.t]);
            migVec = tracks{i}(cellTimeIndex + 1).xyz - tracks{i}(cellTimeIndex).xyz;         
        end
        gradientVectorSum = [0 0 0];       
        for j = 1:length(tracks)
            if j == i
                continue
            end  
            %if other cell is defined at this time point
            otherCellTimeIndex = find([tracks{j}.t] == t);
            if otherCellTimeIndex
                %find vector from this cell to the other
                cell2cellVec = tracks{j}(otherCellTimeIndex).xyz - tracks{i}(cellTimeIndex).xyz;
                %calculate distance
                distance = norm(cell2cellVec);
                normalizedVec = cell2cellVec / distance;
                %scale by 
                %arbitrary scale parameter for numerical stability
                scale = 1 ./ distance.^2;
                scaledVec = scale*normalizedVec;
                gradientVectorSum = gradientVectorSum + scaledVec;
            end  
        end
        tracks{i}(cellTimeIndex).chemokineGrad = gradientVectorSum;
        tracks{i}(cellTimeIndex).migrationVec = migVec;
        normedGradVec = gradientVectorSum / norm(gradientVectorSum);
        normedMigVec = migVec / norm(migVec);
        migGradAngle = acosd(dot(normedGradVec,normedMigVec));    
        tracks{i}(cellTimeIndex).migGradAngle = migGradAngle;
    end
end

tracksWithVecs = tracks;

end

