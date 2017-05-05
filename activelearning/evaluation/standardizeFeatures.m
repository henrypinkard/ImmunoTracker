function [ standardized ] = standardizeFeatures( features )
avg = mean(features);
features = features - repmat(avg,size(features,1),1); 
stddev = std(features);
stddev(stddev == 0) = 1;
standardized = features ./ repmat(stddev, size(features,1),1); 
end

