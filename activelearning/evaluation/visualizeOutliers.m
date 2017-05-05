clear
%%%%% XT Link surface transfer

xtIndex = 0;
javaaddpath ImarisLib.jar

vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(xtIndex);
if (isempty(xImarisApp))
    error('Wrong imaris index');
end
%get file
filename = '/Users/henrypinkard/Google Drive/Research/BIDC/LNImagingProject/activelearningdevelopmentdata/CMTMRCandidates.mat';
189619
surfFile = matfile(filename,'Writable',true);
stats = surfFile.stats;
%get positions of each surface
% xPositions = stats(find(strcmp({stats.Name},'Stitched Position X'))).Values;
% yPositions = stats(find(strcmp({stats.Name},'Stitched Position Y'))).Values;
% zPositions = stats(find(strcmp({stats.Name},'Stitched Position Z'))).Values;
% xyzPositions = [xPositions yPositions zPositions];
% timeIndices = stats(find(strcmp({stats.Name},'Time Index'))).Values;

xSurpass = xImarisApp.GetSurpassScene;

%delete old surface preview
for i = xSurpass.GetNumberOfChildren - 1 :-1: 0
   if (strcmp(char(xSurpass.GetChild(i).GetName),'Preview surface') )
      xSurpass.RemoveChild(xSurpass.GetChild(i)); 
   end
end

xPreviewSurface = xImarisApp.GetFactory.CreateSurfaces;
xPreviewSurface.SetName('Preview surface');
xSurfacesOfInterest = xImarisApp.GetFactory.CreateSurfaces;
xSurpassCam = xImarisApp.GetSurpassCamera;

xSurpass.AddChild(xPreviewSurface,-1);


%make interactive plot of outliers
figure(1)
[~, ~, tCellImarisIndices,nonTCellImarisIndices] = spectralPCAVis([1 0 0]);
dcm_obj = datacursormode;
set(dcm_obj,'UpdateFcn',@DataCursorCallback)

dataPackage = struct('xImarisApp',xImarisApp,'surfFile',surfFile,'xPreviewSurface',...
    xPreviewSurface,'stats',stats,'tCellImarisIndices',tCellImarisIndices,'nonTCellImarisIndices',nonTCellImarisIndices);
guidata(gcf,dataPackage)


