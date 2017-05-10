clear
tab = readtable('MostImportantFeatures.txt');
counts = tab.Var3;
names = tab.Var2;

counts = counts/100;

% 1:10 29:42 85:87 91 97 134:135 morphological
% 11:28 Intensity centers of mass
% 43:84 Intensity features (Unnormalized) 
% 137:178 Intensity features (Spectrally normalized)
% 88:89 136 position in FOV
% 98:133 pairwise
% 134:136 interpolation derived
% 220 NC-ROI # pixels
% 190:204 NC-ROI correlation matrix
% 183:189 NC-ROI mean intensities
% 179:182 NC-ROI intensity projected onto reference spectrum
% 205:219 Global correlation matrix

% 90 92-94 95-96 221-223nothing


indices = {[1:10 29:42 85:87 91 97 134:135], [11:28], [43:84], [137:178], [88:89 136],...
[98:133], [134:136], [190:204],[220],[183:189], [179:182], [205:219]};

labels = {'Morphological', 'Intensity centers of mass', 'Intensity features (Unnormalized)',...
'Intensity features (Spectrally normalized)', 'Position in field of view', 'Pairwise Intensity center of mass distances',...
'Interpolation derived','NC-ROI pixel correlation matrix','NC-ROI # pixels','NC-ROI mean intensities',...
'NC-ROI intensity projected onto reference spectrum','Global pixel correlation matrix'}


indicesToUse = [indices{:}];



h = figure(1)

aHand = axes('parent', h);
hold(aHand, 'on')
colors = hsv(numel(indices));
%randomize colors
colors = colors(randperm(size(colors,1)),:);
barCount = 1;
for categoryIndex = 1:numel(indices)
    dataIndices = indices{categoryIndex};
    for datumIndex = dataIndices
        if datumIndex == 1
            
        end
        bar(barCount, counts(datumIndex), 'parent', aHand, 'facecolor', colors(categoryIndex,:));
        barCount = barCount + 1;  
    end
end


%add category labels
numInCatrgories = cellfun(@numel,indices);
cumnum = cumsum(numInCatrgories);
categorycenters = mean([cumnum; 0 cumnum(1:end-1)]);
set(gca,'XTick',categorycenters)
set(gca,'XTickLabel',labels)
set(gca,'XTickLabelRotation',-60);

ylabel('% of iterations in which feature selected')
hold(aHand, 'off')


