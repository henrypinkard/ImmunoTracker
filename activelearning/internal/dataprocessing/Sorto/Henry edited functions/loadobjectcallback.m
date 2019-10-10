function loadobjectcallback(~, ~, guiSortomato, hPopup)
    %load surfaces from a file and add them into sortomato

    %%Get IUmaris ref
    xImarisApp = getappdata(guiSortomato, 'xImarisApp');
    
    % Get the file
    [filename, pathname] = uigetfile('*.mat');
    if (filename == 0)
        return; 
    end
    
    %load file
    statusbar(guiSortomato, 'Loading file');
    file = matfile(strcat(pathname,filename));
    %add surfaces into imaris
    statusbar(guiSortomato, 'Adding to Imaris');
    
    %add all surfaces from file
    %     xObject = func_addsurfacestosurpass(xImarisApp, file,500);
    
    %create empty surfaces object for later adding
    xObject = xImarisApp.GetFactory.CreateSurfaces;
    xObject.SetName(char(file.name));
    xImarisApp.GetSurpassScene.AddChild(xObject,-1);
    %store matfile for later reading
    setappdata(guiSortomato, 'surfacesFile', file);

    
    statStruct = file.stats;
    %% Find the stats that represent single spots/surfaces and tracks.
    % We store the indices in the stats struct of spot and track stats. This
    % lets us quickly mask to use spot or track stats as selected by users.
    trackStatIdxs = strncmp('Track ', {statStruct.Name}, 6);
    singletStatIdxs = ~trackStatIdxs;
    
    %% Set the data export, graph and stat math callbacks.
    if any(get(guiSortomato, 'Color'))
        fColor = 'k';     
    else
        fColor = 'w';
        
    end % if
    
    set(hPopup, 'ForegroundColor', fColor)
    
    % Update the export, graph and stat math callbacks.
    pushExportStats = findobj(guiSortomato, 'Tag', 'pushExportStats');
    set(pushExportStats, 'Callback', {@pushexportstatscallback, xImarisApp, xObject, guiSortomato})
    
    pushGraph = findobj(guiSortomato, 'Tag', 'pushGraph');
    set(pushGraph, 'Callback', {@sortomatograph, statStruct, guiSortomato})
    
    pushGraph3 = findobj(guiSortomato, 'Tag', 'pushGraph3');
    set(pushGraph3, 'Callback', {@sortomatograph3, statStruct, guiSortomato})
    
    pushStatMath = findobj(guiSortomato, 'Tag', 'pushStatMath');
    set(pushStatMath, 'Callback', {@sortomatostatmath, statStruct, guiSortomato})
    
    %% Update the list of Surpass objects in the base GUI.
    surpassObjects = xtgetsporfaces(xImarisApp);

    % Get the base GUI's objects popup.
    popupObjects = findobj(guiSortomato, 'Tag', 'popupObjects');
    set(popupObjects, 'String', {surpassObjects.Name})

    % Store the updated object list in the the listboxes app data.
    setappdata(popupObjects, 'surpassObjects', surpassObjects)

    
    %% Store the statistics data and selected object as appdata.
    setappdata(guiSortomato, 'statStruct', statStruct);
    setappdata(guiSortomato, 'trackStatIdxs', trackStatIdxs);
    setappdata(guiSortomato, 'singletStatIdxs', singletStatIdxs);
    setappdata(hPopup, 'xObject', xObject)
    
    %% Reset the status bar.
    statusbar(guiSortomato, '');
end % popupobjectscallback