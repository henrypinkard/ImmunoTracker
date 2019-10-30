clear

vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(0);
if (isempty(xImarisApp))
    error('Wrong imaris index');
end

xDataset = xImarisApp.GetDataSet;
xDataset.SetSizeC(7);
timeIndex = 0;
xcr1Vec = getReferenceVector('xcr1')';
slices = zeros(6, xDataset.GetSizeX, xDataset.GetSizeY, 'double'); 
projected = {};
for z = 0:xDataset.GetSizeZ - 1
    fprintf('z %i\n', z+1);
    for c  = 0:5
        fprintf('getting channel %i\n', c+1);
        raw = xDataset.GetDataSliceBytes(z, c, timeIndex);
        slices(c+1, :, :) = double(reshape(typecast(raw(:), 'uint8'), size(raw)));
    end
    projSlice = squeeze(sum(slices .* xcr1Vec));
    projected{z + 1} = projSlice;
end
volume = uint8(cat(3, projected{:}));

delta = 20;
for zDest = 1:10:size(volume, 3)
    fprintf('Sending in data slice %i\n', zDest)
    xDataset.SetDataSubVolumeBytes(volume(:, :, zDest:min(zDest + delta, end)), 0, 0, zDest-1, c + 1, timeIndex);
end

