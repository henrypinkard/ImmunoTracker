function xtTransferImageData( img )

xtIndex = 0;
javaaddpath('./ImarisLib.jar')
vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(xtIndex);
if (isempty(xImarisApp))
    fprintf('Wrong imaris index');
    return;
end

xFactory = xImarisApp.GetFactory();
xDataset = xFactory.CreateDataSet();

xDataset.Create( Imaris.tType.eTypeUInt8,int32(size(img,1)),int32(size(img,2)),int32(size(img,3)),int32(6),int32(1))
for channel = 1:6
    xDataset.SetDataVolumeBytes(img(:,:,:,channel), channel-1, 0);
end
%calibrate
pixelSizeXY = 0.350819;
pixelSizeZ = 4.5;
xDataset.SetExtendMaxX(size(img,1)*pixelSizeXY)
xDataset.SetExtendMaxY(size(img,2)*pixelSizeXY)
xDataset.SetExtendMaxZ(size(img,3)*pixelSizeZ)
%set colors
xDataset.SetChannelColorRGBA(0,15671199);
xDataset.SetChannelColorRGBA(1,10053120);
xDataset.SetChannelColorRGBA(2,65280);
xDataset.SetChannelColorRGBA(3,65535);
xDataset.SetChannelColorRGBA(4,255);
xDataset.SetChannelColorRGBA(5,13369548);
xImarisApp.SetDataSet(xDataset)
%set contrast
xDataset.SetChannelRange(0,10,250)
xDataset.SetChannelRange(1,10,100)
xDataset.SetChannelRange(2,12,100)
xDataset.SetChannelRange(3,15,250)
xDataset.SetChannelRange(4,10,100)
xDataset.SetChannelRange(5,10,100)


end

