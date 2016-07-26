function [  ] = createDistanceMaps(  )
    distanceTransforms = cell(51,1);
    file = matfile('Left LN_2_Surfaces 1 surface masks.mat');
    pixXY = file.pixelSizeXY *10;
    pixZ = file.pixelSizeZ;
    tic
    for timeIndex = 0:50
        toc
        fprintf('time index %o\n',timeIndex);
        maskIndex = timeIndex;
        if timeIndex == 0
            maskIndex = 1; %no first mask
        end
        surfaceMask = file.mask(:,:,:,maskIndex);
        %pad edges to make divisible by 10
        surfaceMask = [surfaceMask; zeros(6, 4270, 89)];
        %downsample in xy to make distance transform easier
        c=reshape(surfaceMask,[10 389 10 427 89]);
        c=sum(c,1);
        c=sum(c,3);
        dsMask=reshape(c,[389 427 89]);
        dsMask(dsMask >=1) = 1;
        transform = bwdistsc(dsMask,[pixXY pixXY pixZ]);
        distanceTransforms{timeIndex+1} = transform;
    end
    
    save('DistanceTransform','distanceTransforms','pixXY','pixZ');
end

