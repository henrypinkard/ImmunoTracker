% Calculate distance to selected surface, add into statStruct, resave
% Workflow: use this script in between saving and sorting
function [] = addcustomstats()

%get file corresponding to surfaces
[filename, pathname] = uigetfile('*.mat');
if (filename == 0)
    return;
end
surfFile = matfile(strcat(pathname, filename),'Writable',true);

%get stat struct
stats = surfFile.stats;

vIdx = find(ismember({stats.Name},'Intensity Mean - Channel 1'));
bIdx = find(ismember({stats.Name},'Intensity Mean - Channel 2'));
gIdx = find(ismember({stats.Name},'Intensity Mean - Channel 3'));
yIdx = find(ismember({stats.Name},'Intensity Mean - Channel 4'));
rIdx = find(ismember({stats.Name},'Intensity Mean - Channel 5'));
frIdx = find(ismember({stats.Name},'Intensity Mean - Channel 6'));

stats(vIdx).Name = 'Intensity Mean: Violet';
stats(bIdx).Name = 'Intensity Mean: Blue';
stats(gIdx).Name = 'Intensity Mean: Green';
stats(yIdx).Name = 'Intensity Mean: Yellow';
stats(rIdx).Name = 'Intensity Mean: Red';
stats(frIdx).Name = 'Intensity Mean: Far red';

%add ratio stats
singlestruct = @(name, values) struct('Ids',stats(1).Ids,'Name',name,'Values',values,'Units','');

stats(length(stats) + 1) = singlestruct('Ratio: Blue / Violet',stats(bIdx).Values ./ stats(vIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Green / Violet',stats(gIdx).Values ./ stats(vIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Yellow / Violet',stats(yIdx).Values ./ stats(vIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Red / Violet',stats(rIdx).Values ./ stats(vIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Far Red / Violet',stats(frIdx).Values ./ stats(vIdx).Values);

stats(length(stats) + 1) = singlestruct('Ratio: Green / Blue',stats(gIdx).Values ./ stats(bIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Yellow / Blue',stats(yIdx).Values ./ stats(bIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Red / Blue',stats(rIdx).Values ./ stats(bIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Far Red / Blue',stats(frIdx).Values ./ stats(bIdx).Values);

stats(length(stats) + 1) = singlestruct('Ratio: Yellow / Green',stats(yIdx).Values ./ stats(gIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Red / Green',stats(rIdx).Values ./ stats(gIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Far Red / Green',stats(frIdx).Values ./ stats(gIdx).Values);

stats(length(stats) + 1) = singlestruct('Ratio: Red / Yellow',stats(rIdx).Values ./ stats(yIdx).Values);
stats(length(stats) + 1) = singlestruct('Ratio: Far Red / Yellow',stats(frIdx).Values ./ stats(yIdx).Values);

stats(length(stats) + 1) = singlestruct('Ratio: Far Red / Red',stats(frIdx).Values ./ stats(rIdx).Values);


%sort into alphabetical order
[~, statOrder] = sort({stats.Name});
stats = stats(statOrder);
%resave
surfFile.stats = stats;
end
