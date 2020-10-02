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

load('/Users/henrypinkard/Desktop/imaris_analysis/4445_spots.mat')

spots = factory.CreateSpots;
spots.SetName('XCR1_clustering')
spots.Set(double(XCR1_clustering_coords(:, 1:3)), double(XCR1_clustering_time_indices(:)), double(XCR1_clustering_coords(:, 4)))
xSurpass.AddChild(spots, -1);
spots.SetTrackEdges(double(XCR1_clustering_edges))


spots = factory.CreateSpots;
spots.SetName('GFP')
spots.Set(double(GFP_coords(:, 1:3)), double(GFP_time_indices(:)), double(GFP_coords(:, 4)))
xSurpass.AddChild(spots, -1);
spots.SetTrackEdges(double(GFP_edges))

% spots = factory.CreateSpots;
% spots.SetName('RFP')
% spots.Set(double(RFP_coords(:, 1:3)), double(RFP_time_indices(:)), double(RFP_coords(:, 4)))
% xSurpass.AddChild(spots, -1);
%  spots.SetTrackEdges(double(RFP_edges))

spots = factory.CreateSpots;
spots.SetName('VPD')
spots.Set(double(VPD_coords(:, 1:3)), double(VPD_time_indices(:)), double(VPD_coords(:, 4)))
xSurpass.AddChild(spots, -1);
 spots.SetTrackEdges(double(VPD_edges))

