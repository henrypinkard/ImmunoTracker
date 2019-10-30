function [newFeatures, newFeatureNames] = calcCOMFeatures(features, featureNames )

 cohmxIndex = find(strcmp('Center of Homogeneous Mass X',featureNames));
 cohmyIndex = find(strcmp('Center of Homogeneous Mass Y',featureNames));
 cohmzIndex = find(strcmp('Center of Homogeneous Mass Z',featureNames));
 coimx1Index = find(strcmp('Center of Image Mass X - Channel 1',featureNames));
 coimx2Index = find(strcmp('Center of Image Mass X - Channel 2',featureNames));
 coimx3Index = find(strcmp('Center of Image Mass X - Channel 3',featureNames));
 coimx4Index = find(strcmp('Center of Image Mass X - Channel 4',featureNames));
 coimx5Index = find(strcmp('Center of Image Mass X - Channel 5',featureNames));
 coimx6Index = find(strcmp('Center of Image Mass X - Channel 6',featureNames));
 coimy1Index = find(strcmp('Center of Image Mass Y - Channel 1',featureNames));
 coimy2Index = find(strcmp('Center of Image Mass Y - Channel 2',featureNames));
 coimy3Index = find(strcmp('Center of Image Mass Y - Channel 3',featureNames));
 coimy4Index = find(strcmp('Center of Image Mass Y - Channel 4',featureNames));
 coimy5Index = find(strcmp('Center of Image Mass Y - Channel 5',featureNames));
 coimy6Index = find(strcmp('Center of Image Mass Y - Channel 6',featureNames));
 coimz1Index = find(strcmp('Center of Image Mass Z - Channel 1',featureNames));
 coimz2Index = find(strcmp('Center of Image Mass Z - Channel 2',featureNames));
 coimz3Index = find(strcmp('Center of Image Mass Z - Channel 3',featureNames));
 coimz4Index = find(strcmp('Center of Image Mass Z - Channel 4',featureNames));
 coimz5Index = find(strcmp('Center of Image Mass Z - Channel 5',featureNames));
 coimz6Index = find(strcmp('Center of Image Mass Z - Channel 6',featureNames));
 
 
%Center of homogenous mass XYZ
cohm = features(:,[cohmxIndex cohmyIndex cohmzIndex]);
%Intensity weighted mass offset
coim = features(:,[coimx1Index coimx2Index coimx3Index coimx4Index coimx5Index coimx6Index...
    coimy1Index coimy2Index coimy3Index coimy4Index coimy5Index coimy6Index...
    coimz1Index coimz2Index coimz3Index coimz4Index coimz5Index coimz6Index]);
%split by channel
coim1 = coim(:,1:6:end);
coim2 = coim(:,2:6:end);
coim3 = coim(:,3:6:end);
coim4 = coim(:,4:6:end);
coim5 = coim(:,5:6:end);
coim6 = coim(:,6:6:end);

intenistyWeightOffset = zeros(size(features,1),6);
%distance for each channel
intenistyWeightOffset(:,1) = sqrt( sum( (coim1 - cohm).^2, 2) );
intenistyWeightOffset(:,2) = sqrt( sum( (coim2 - cohm).^2, 2) );
intenistyWeightOffset(:,3) = sqrt( sum( (coim3 - cohm).^2, 2) );
intenistyWeightOffset(:,4) = sqrt( sum( (coim4 - cohm).^2, 2) );
intenistyWeightOffset(:,5) = sqrt( sum( (coim5 - cohm).^2, 2) );
intenistyWeightOffset(:,6) = sqrt( sum( (coim6 - cohm).^2, 2) );

newFeatureNames = {'Intensity weight offset Ch1' 'Intensity weight offset Ch2'...
    'Intensity weight offset Ch3' 'Intensity weight offset Ch4'...
    'Intensity weight offset Ch5' 'Intensity weight offset Ch6'};
newFeatures = [intenistyWeightOffset];

%pairwise center of image mass differences
for i = 1:6
    for j = 1+1:6
        %i.j pairwise distance
        newFeatureNames = {newFeatureNames{:} sprintf('Center of Image Mass pairwise distance %i_%i',i,j)}';
        newFeatures = [newFeatures eval(sprintf('sqrt( sum( (coim%i - coim%i).^2, 2) );',i,j))];
    end
end

end

