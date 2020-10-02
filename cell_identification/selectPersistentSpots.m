clear
cd(fileparts(mfilename('fullpath')))
xtIndex = 0;
javaaddpath('./ImarisLib.jar')
vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(xtIndex);
if (isempty(xImarisApp))
    error('Wrong imaris index');
end
factory = xImarisApp.GetFactory;

xSurpass = xImarisApp.GetSurpassScene;

load('/Users/henrypinkard/Desktop/imaris_analysis/48_49_fusion_spots.mat')

spots = factory.CreateSpots;
spots.SetName('XCR1')
spots.Set(double(XCR1_coords(:, 1:3)), double(XCR1_time_indices(:)), double(XCR1_coords(:, 4)))
xSurpass.AddChild(spots, -1);
spots.SetTrackEdges(double(XCR1_edges))
