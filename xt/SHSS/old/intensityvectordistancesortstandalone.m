clear
clc
%TODO: subtract channel offsets from new data
%TODO: maybe find minimum distance to any one surface in the group?
%TODO: interactive module where you find one, then others get added at same
%time point that are similar, then you slect which of those are good, and
%repeat so it learns from what you show it

xImaris = xtconnectimaris(0);
surfaces = xImaris.GetFactory.ToSurfaces( xImaris.GetSurpassSelection);
stats = xtgetstats(xImaris, surfaces, 'ID', 'ReturnUnits', 1);

vIdx = find(ismember({stats.Name},'Intensity Mean - Channel 1'));
bIdx = find(ismember({stats.Name},'Intensity Mean - Channel 2'));
gIdx = find(ismember({stats.Name},'Intensity Mean - Channel 3'));
yIdx = find(ismember({stats.Name},'Intensity Mean - Channel 4'));
rIdx = find(ismember({stats.Name},'Intensity Mean - Channel 5'));
frIdx = find(ismember({stats.Name},'Intensity Mean - Channel 6'));

%create 6 dimensional vector with intensity stats
% intensityMat = [stats(vIdx).Values, stats(bIdx).Values, stats(gIdx).Values, stats(yIdx).Values, stats(rIdx).Values, stats(frIdx).Values];
%exclude red and far red, because interactions with MTMG DCs screw things
%up
intensityMat = [stats(vIdx).Values, stats(bIdx).Values, stats(gIdx).Values, stats(yIdx).Values, stats(yIdx).Values, stats(frIdx).Values];

intensityVecs = mat2cell(intensityMat,ones(size(intensityMat,1),1),size(intensityMat,2));
%normalize to a length of 1
norms = cellfun(@norm,intensityVecs);
normalizedVecs = mat2cell(bsxfun(@rdivide,intensityMat,norms),ones(size(intensityMat,1),1),size(intensityMat,2));
%normalized vectors give coordinates for each element on the surface of a
%six dimensional hypersphere

%try finding closest to a known cell
imarisIndices = stats(1).Ids;
ids = surfaces.GetSelectedIds;
indices = find( ismember(imarisIndices,double(ids)));
referencePosition = median(cell2mat(normalizedVecs(indices)),1); %median is more resistant to outliers

distances = cellfun(@(v) norm(v - referencePosition),normalizedVecs);
%show histogram
figure(1)
hist(distances,500);
% ask for user input of cutoff
cutoff = input('distance cutoff? ')
[~, i] = sort(distances);
sortedIds = imarisIndices(i); 

%create surfaces and add
surfaces.SetSelectedIndices(double(sortedIds(1:sum(distances <= cutoff))));

