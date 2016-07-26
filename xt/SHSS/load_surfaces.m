%Read surfaces from .mat file and add them
function [] = load_surfaces(imarisIndex)

batchSize = 1000;

vImarisLib = ImarisLib;
imaris = vImarisLib.GetApplication(imarisIndex);
if (isempty(imaris))
    msgbox('Wrong imaris index');
    return;
end


%get file
[filename, pathname] = uigetfile('*.mat');
if (filename == 0)
    return;
end

surfFile = matfile(strcat(pathname,filename));
func_addsurfacestosurpass(imaris,surfFile, batchSize);
end
