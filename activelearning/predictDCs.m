clear
load('DCSelectionsAndTimeIndices');
load('DCFeaturesAndLabels');

firstFourTimeIndices = find(timeIndices <= 3) - 1;
dcsFirstFourTPII = intersect(firstFourTimeIndices,selectedImarisIDs);
nonDCsFirstFourTPII = setdiff(firstFourTimeIndices,selectedImarisIDs);

dcsLaterTPII = setdiff(intersect(selectedImarisIDs,imarisIndices),firstFourTimeIndices);


%Predict based on all labeled data
labelledDCMask = ismember(imarisIndices, dcsFirstFourTPII);
labelledNotDCMask = ismember(imarisIndices, nonDCsFirstFourTPII);
trainDataMask = labelledDCMask | labelledNotDCMask;

labels = zeros(size(labelledDCMask,1),2);
labels(labelledDCMask,1) = 1;
labels(labelledNotDCMask,2) = 1;
labels = labels(trainDataMask,:);

%Determine number of hiddens by cross validation
% nHiddens = [1 5 10 15 20 50 100 120]
% perf = zeros(size(nHiddens));
% for index = 1:length(nHiddens)
%     nHidden = nHiddens(index)
%     net = patternnet(nHidden,'trainscg','crossentropy' );
%     x = features(dataToUseMask,:)';
%     t = labels';
%     [net,tr] = train(net,x,t );
%     
%     %evaluate on test data
%     testX = x(:,tr.testInd);
%     testT = t(:,tr.testInd);
%     testY = net(testX);
%     % testClasses = testY > 0.5
%     [c,cm] = confusion(testT,testY);
%     perf(index) = c;
% end
% plot(nHiddens,perf,'o-')
% 
% plotconfusion(testT,testY)
% plotroc(testT,testY)


%%train neual net on all data and predict unseens
%best number of hiddens = 20;
net = patternnet(20,'trainscg','crossentropy' );
net.divideParam.testRatio = 0;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
x = features(trainDataMask,:)';
t = labels';
[net,tr] = train(net,x,t);

%predict remaining time points
pred = net( features(~trainDataMask,:)' )';

cutoff = 0.35;
predLabelDC = find(pred(:,1) > cutoff);
predSurfacesII = imarisIndices(~trainDataMask);
predDCsImarisIndices = predSurfacesII(predLabelDC);

% addDCSetToImaris(predDCsImarisIndices);

%k means to split into three
allDCImarisIndics = [predDCsImarisIndices; dcsFirstFourTPII];
%find indices in feture vector
featureMatMask = ismember(imarisIndices,allDCImarisIndics);

%take mean intensity
unstandardized = load('DCFeaturesAndLabelsUnstandardized','features');
spectra = unstandardized.features(featureMatMask,114:119);


green =   [0.0701    0.0660    0.9262    0.3365    0.1170    0.0777]; %green DC
greenPink =   [  0.0506    0.0347    0.8640    0.3520    0.1245    0.3322]; %green pink DC
autofluor =   [ 0.0844    0.1937    0.7024    0.5647    0.3643 0.1018];   %autofluor DC
greenRed = [0.0426    0.0297    0.6433    0.3601    0.6675 0.0909]; %green red DC

numDCs = size(spectra,1);
greenDist = dot(spectra, repmat(green,numDCs,1),2);
gpDist = dot(spectra, repmat(greenPink,numDCs,1),2);
grDist = dot(spectra, repmat(greenRed,numDCs,1),2);
afDist = dot(spectra, repmat(autofluor,numDCs,1),2);
allDists = [greenDist grDist gpDist afDist ];
[~, closestColorIndex] = max(allDists,[],2);

%visualize
idx = kmeans(spectra,4);
[coeff, score, latent, tsquared, explained] = pca(spectra);
scatter(score(:,1),score(:,2),[45],idx,'filled')



%add three different sets to iamris corresponding to different colors
reshuffledII = imarisIndices(featureMatMask);

%add each cluster to imaris seperately
for clusterIndex = 1:max(idx)
   addDCSetToImaris(reshuffledII(idx == clusterIndex));
end

for clusterIndex = 1:max(closestColorIndex)
   addDCSetToImaris(reshuffledII(closestColorIndex == clusterIndex));
end

