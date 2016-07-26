function [ pixelData ] = readRawMagellan( magellanDataset, surfaceMask, offset, timeIndex )
%offset is in in 0 indexed pixels
%timeIndexStarts with 0
unsign = @(arr) uint8(bitset(arr,8,0)) + uint8(128*(arr < 0));

%Read raw pixels from magellan
%take stiched pixels instead of decoding position indices
pixelData = zeros(size(surfaceMask,1),size(surfaceMask,2),size(surfaceMask,3),6, 'uint8'); 
for channel = 0:5
    for relativeSlice = 0:size(surfaceMask,3)-1
        slice = offset(3) + relativeSlice;
        ti = magellanDataset.getImageForDisplay(channel, slice, timeIndex, 0, offset(1), offset(2),...
            size(pixelData,1), size(pixelData,2));
        pixels = reshape(unsign(ti.pix),size(pixelData,1),size(pixelData,2));  
       
        pixelData(:,:,relativeSlice+1,channel+1) = pixels;
    end
end

a = 10;

end

