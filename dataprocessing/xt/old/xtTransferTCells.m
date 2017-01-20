xtIndex = 0;
javaaddpath('./ImarisLib.jar')
vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(xtIndex);
if (isempty(xImarisApp))
    error('Wrong imaris index');
end

xSurpass = xImarisApp.GetSurpassScene;
xSurface = xImarisApp.GetFactory.CreateSurfaces;
xSurface.SetName('T Cells');
xSurpass.AddChild(xSurface,-1);

data = load('CMTMRFeaturesAndLabels.mat','imarisIndices','labelledTCell');
ii = data.imarisIndices;
tCellIndices = data.labelledTCell + 1;
filename = '/Users/henrypinkard/Desktop/LNData/CMTMRCandidates.mat';
surfFile = matfile(filename);
func_addsurfacestosurpass(xImarisApp,surfFile,1,xSurface,ii(tCellIndices));
