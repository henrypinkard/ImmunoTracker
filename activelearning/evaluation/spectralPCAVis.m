 function [tCellScores, nonTCellScores, tCellImarisIndices,nonTCellImarisIndices] = spectralPCAVis(tCellColors)
load('PreprocessedCMTMRData.mat')
tCellImarisIndices = imarisIndices(labelledTCell+1);
nonTCellImarisIndices = imarisIndices(labelledNotTCell+1);

intensityMeans = features(:,55:60);

%spectral normalization
channelOffsets = [8 12 16 12 10 8];
%subtract offsets
intensityMeans = intensityMeans - repmat(channelOffsets,size(intensityMeans,1),1);
%Make strictly positive
intensityMeans(intensityMeans < 0) = 0; 
%calculate spectral magnitudes
spectralMagnitudeMean = sqrt(sum(intensityMeans.^2,2));
%remove 0 magnitdudes
spectralMagnitudeMean(spectralMagnitudeMean == 0) = 1;
%spectral normalization
normalizedSpecMeans = intensityMeans ./ repmat(spectralMagnitudeMean,1,6);



spectralMetric4pca = normalizedSpecMeans;

tCellSpecMetric = spectralMetric4pca(labelledTCell+1,:);
nonTCellSpecMetric = spectralMetric4pca(labelledNotTCell+1,:);
unlabelledSpecMetric = spectralMetric4pca(unlabelled+1,:);

stat = ismember(1:size(features,1),labelledTCell+1); %binary mark T cells

% tCellColors = stat(labelledTCell+1);
nonTCellColors = stat(labelledNotTCell+1);

%find principal axes for just T cells
[coeff, score, latent, tsquared, explained] = pca(spectralMetric4pca(labelledTCell+1,:));
% [coeff, score, latent, tsquared, explained] = pca(spectralMetric4pca);


tCellScores = tCellSpecMetric*coeff;
nonTCellScores = nonTCellSpecMetric*coeff;
unlabelledScores = unlabelledSpecMetric*coeff;

scatter(nonTCellScores(:,1),nonTCellScores(:,2),[25],[0 0.5 0.5],'filled','MarkerEdgeColor','none')
colormap inferno
alpha(0.1)
hold on
scatter(tCellScores(:,1),tCellScores(:,2),[150],tCellColors,'filled','MarkerEdgeColor',[0 0 0])
hold off
% legend('Not T Cell','T Cell')
xlabel('Spectral PC1')
ylabel('Spectral PC2')
% setFontsAndLines()
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
h = colorbar;
ylabel(h,'Probability of misclassification')

end

