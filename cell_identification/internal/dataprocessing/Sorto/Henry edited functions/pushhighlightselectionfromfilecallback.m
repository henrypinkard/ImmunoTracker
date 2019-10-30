function pushhighlightselectionfromfilecallback(~, ~, xObject, axesGraph, sortomatoGraph)
   
    guiSortomato = getappdata(sortomatoGraph, 'guiSortomato');
       
    %get the mat file object       
    surfFile = getappdata(guiSortomato,'surfacesFile');
    iSelection = surfFile.selectedSurfaces;
    
%     %testing vector distance sort
%     sortedIndices = intensityvectordistancesort(surfFile);
%     iSelection = sortedIndices(1:500);
%     %clear old ones
%     xObject.RemoveAllSurfaces;
%     func_addsurfacestosurpass([], surfFile, [], xObject, iSelection); 
    
    
    %get statStruct of figure to convert imaris indices to data indices
    statStruct = getappdata(sortomatoGraph,'statStruct');
    xIDs = double(statStruct(1).Ids);
    
    rgnColorMask = ismember(xIDs, iSelection);
    xColor = rgb32bittotriplet(xObject.GetColorRGBA);

    
    hScatter = getappdata(axesGraph, 'hScatter');
    xData = getappdata(axesGraph, 'xData');
    yData = getappdata(axesGraph, 'yData');

    %draw nomral objects first then sleected ones on top
    set(hScatter(1), ...
        'MarkerFaceColor', 1 - xColor, ...
        'XData', xData(rgnColorMask), ...
        'YData', yData(rgnColorMask))

    delete(findobj(axesGraph, 'Tag', 'hScatter2'))
    hScatter(2) = line(...
        'LineStyle', 'none', ...
        'Marker', 'd', ...
        'MarkerEdgeColor', 'none', ...
        'MarkerFaceColor', xColor, ...
        'MarkerSize', 3, ...
        'Parent', axesGraph, ...
        'Tag', 'hScatter2', ...
        'XData', xData(~rgnColorMask), ...
        'YData', yData(~rgnColorMask));
    uistack(hScatter, 'bottom')

    %% Store the region color mask and scatter handle array.
    setappdata(axesGraph, 'rgnColorMask', rgnColorMask)
    setappdata(axesGraph, 'hScatter', hScatter);
end % pushhighlightselectioncallback