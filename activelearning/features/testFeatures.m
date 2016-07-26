 function [tCellScores, nonTCellScores, tCellImarisIndices,nonTCellImarisIndices] = testFeatures()
load('FeaturesAndLabelsTop2xNC.mat')

tCellImarisIndices = imarisIndices(labelledTCell+1);
nonTCellImarisIndices = imarisIndices(labelledNotTCell+1);
normalizedSpecMeans = features(:,114:119);

spectralMetric4pca = normalizedSpecMeans;

tCellSpecMetric = spectralMetric4pca(labelledTCell+1,:);
nonTCellSpecMetric = spectralMetric4pca(labelledNotTCell+1,:);
unlabelledSpecMetric = spectralMetric4pca(unlabelled+1,:);

% find spectral median t cell
tCellNormalizedSpecMeans = normalizedSpecMeans(labelledTCell+1,:);
medianTCellSpecMean = median(tCellNormalizedSpecMeans,1);
%renormalize
medianTCellSpecMean = medianTCellSpecMean / norm(medianTCellSpecMean);
%score by spectral distance to median
specDist = sum(normalizedSpecMeans.*repmat(medianTCellSpecMean,size(normalizedSpecMeans,1),1),2);
% specDist = sum(normalizedSpecMedians.*repmat(medianTCellSpecMean,size(normalizedSpecMedians,1),1),2);


%color codings
% stat = features(:,135); %vertical distance
% stat = features(:,134); %distance to cortex
% stat = features(:,126); %pairwise coim distance ch #
% stat = features(:,100); %Intenisty weight offset ch #
% stat = features(:,87); %numVoxels
% stat = specDist.^0.8;
% stat = log(features(:,138)); %projected intensity sum
% stat = features(:,153); %global green red corr mat
% stat = (features(:,167)); %ROI green red corr mat


stat = features(:,141); %projected intensity avg
% stat = features(:,139); %percent of pixels masked
% stat = ismember(1:size(features,1),labelledTCell+1); %binary mark T cells



figure(2)
hist([stat(labelledTCell+1); stat(labelledNotTCell+1)],200)

tCellColors = stat(labelledTCell+1);
nonTCellColors = stat(labelledNotTCell+1);

%find principal axes for just T cells
[coeff, score, latent, tsquared, explained] = pca(spectralMetric4pca(labelledTCell+1,:));
% [coeff, score, latent, tsquared, explained] = pca(spectralMetric4pca);


tCellScores = tCellSpecMetric*coeff;
nonTCellScores = nonTCellSpecMetric*coeff;
unlabelledScores = unlabelledSpecMetric*coeff;

figure(1)
scatter(nonTCellScores(:,1),nonTCellScores(:,2),[25],nonTCellColors,'filled','MarkerEdgeColor','none')
colormap viridis
alpha(0.8)
hold on
scatter(tCellScores(:,1),tCellScores(:,2),[65],tCellColors,'filled','MarkerEdgeColor',[0 0 0])
hold off
% legend('Not T Cell','T Cell')
xlabel('Spectral principal component 1')
ylabel('Spectral principal component 2')
setFontsAndLines()
hold on
%draw color dircetions
arrowCoeff = coeff*0.4;
quiver( 0.6,0.6,arrowCoeff(1,1),arrowCoeff(1,2),0,'color',[.3 .1 .7],'linewidth', 3)
quiver( 0.6,0.6,arrowCoeff(2,1),arrowCoeff(2,2),0,'color',[0 0 1],'linewidth', 3)
quiver( 0.6,0.6,arrowCoeff(3,1),arrowCoeff(3,2),0,'color',[0 1 0],'linewidth', 3)
quiver( 0.6,0.6,arrowCoeff(4,1),arrowCoeff(4,2),0,'color',[1 1 0],'linewidth', 3)
quiver( 0.6,0.6,arrowCoeff(5,1),arrowCoeff(5,2),0,'color',[1 0 0],'linewidth', 3)
quiver( 0.6,0.6,arrowCoeff(6,1),arrowCoeff(6,2),0,'color',[1 0 0.5],'linewidth', 3)
hold off
colorbar

end

