function [] = timeprojectsurfaces()
%TODO: experiment with this value
spot_radius = 20;


%TODO: input pixels sizes
pixelSizeXY = 0.758;
pixelSizeZ = 2.06;
pathname = 'C:\Users\hpinkard\Desktop\';
name = 'data';

%TODO: get width and height and depth of data set (pixels) using XT
width = 1024;
height = 1024;
numSlices = 50;

%TODO: get XYZ coodrinates of cells of interest using XT
%surfaceCenters = rand(100,3);
surfaceCenters = importdata('greens_pos.txt');

surfaceCenters = [surfaceCenters(:,1)*width*pixelSizeXY...
    surfaceCenters(:,2)*height*pixelSizeXY surfaceCenters(:,3)*numSlices*pixelSizeZ];




%Matlab librarypath.txt must be set before starting Matlab to include the
%location of dll files
javaaddpath C:\Users\hpinkard\Dropbox\Henry\Code\Libraries\jhdf5.jar;
javaaddpath C:\Users\hpinkard\Dropbox\Henry\Code\Libraries\jhdf5obj.jar;
javaaddpath C:\Users\hpinkard\Dropbox\Henry\Code\Libraries\jhdfobj.jar;
javaaddpath C:\Users\hpinkard\Dropbox\Henry\Code\Builds\Imaris_writer.jar;
javaaddpath C:\Users\hpinkard\Dropbox\Henry\Code\Libraries\jhdf.dll;
javaaddpath C:\Users\hpinkard\Dropbox\Henry\Code\Libraries\jhdf5.dll;




%add 1 to pixels to account for Matlabs one based indexing
surfaceCenterPixelCoords = floor(surfaceCenters ./ repmat([pixelSizeXY pixelSizeXY pixelSizeZ],size(surfaceCenters,1),1)) + 1;

prefix = strcat(name,sprintf('_%d um  projection',spot_radius));
imarisWriter = HDF.ImarisWriter(pathname,prefix,width,height,numSlices,1,1,pixelSizeXY,pixelSizeZ,[]);


%create empty pixel data
pixels = zeros(width, height, numSlices, 'uint8');

pixelRadiusXY = spot_radius / pixelSizeXY;
pixelRadiusZ = spot_radius / pixelSizeZ;
for i = 1:size(surfaceCenterPixelCoords,1)
    %add to pixel values in time projection channel based on coordinates of spot
    for x = max(1,surfaceCenterPixelCoords(i,1) - pixelRadiusXY):min(size(pixels,1), surfaceCenterPixelCoords(i,1) + pixelRadiusXY)
        for y = max(1,surfaceCenterPixelCoords(i,2) - pixelRadiusXY): min(size(pixels,2), surfaceCenterPixelCoords(i,2) + pixelRadiusXY)
            for z = max(1,surfaceCenterPixelCoords(i,3) - pixelRadiusZ):min(size(pixels,3), surfaceCenterPixelCoords(i,3) + pixelRadiusZ)
                %iterating through a 3D rectangle of all possible pixels,
                %increment only those that lie within inscribed ellipsoid
                %subtract one and multiply by pixel size to get um position
                distance = sqrt(sum([pixelSizeXY*((x-1) - surfaceCenterPixelCoords(i,1)) pixelSizeXY*((y-1) - surfaceCenterPixelCoords(i,2))...
                    pixelSizeZ*((z-1) - surfaceCenterPixelCoords(i,3))].^2));
                if  distance < spot_radius
                    indices = round([x y z]);
                    pixels(indices(1), indices(2), indices(3)) = pixels(indices(1), indices(2), indices(3)) + 1;
                end
            end
        end
    end
    fprintf('surface: %d of %d\n',i,size(surfaceCenterPixelCoords,1));
end

%write pixels to tiff storage
for s = 0:numSlices-1
    %add slices
    fprintf('writing slice %i\n',s);
    imarisWriter.addImage(reshape(pixels(:,:,s+1),1,width*height),s,0,0,[],'');
end

%close hdfWriter
imarisWriter.close();
end
