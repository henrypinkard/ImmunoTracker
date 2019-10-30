function [xSpots] = addSpotsToSurpass(xImarisApp, spotIndicesToAdd, name, spotCenters, timeIndices)
%surfIndicesToAdd is 0 index imarisIndices
spotIndicesToAdd = spotIndicesToAdd + 1;

h = figure(1);

xSurpass = xImarisApp.GetSurpassScene;
xPopulationSpots = xImarisApp.GetFactory.CreateSpots;
xPopulationSpots.SetName(name);
xSurpass.AddChild(xPopulationSpots,-1);

        
xPopulationSpots.Set(spotCenters(spotIndicesToAdd, :), timeIndices(spotIndicesToAdd), 6 * ones(size(spotIndicesToAdd)))

%check if file has track edges
% if (any(strcmp('trackEdges',who(file))))
%     xSpots.SetTrackEdges(file.trackEdges);
% end

end
