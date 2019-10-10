function sortomatosurfaceintensity(~, ~, hSortomatoBase)
    % SORTOMATOSURFACEINTENSITY Calculate intensities in surface and
    % interior voxels
    %   Detailed explanation goes here
    %
    %  ©2010-2013, P. Beemiller. Licensed under a Creative Commmons Attribution
    %  license. Please see: http://creativecommons.org/licenses/by/3.0/
    %
    
    %% Check for an already-running GUI.
    guiChildren = getappdata(hSortomatoBase, 'guiChildren');
    
    if ~isempty(guiChildren)
        guiSurfaceIntCalc = findobj(guiChildren, 'Tag', 'guiSurfaceIntCalc');
        
        if ~isempty(guiSurfaceIntCalc)
            figure(guiSurfaceIntCalc)
            return
        end % if
    end % if
    
    %% Get the Surpass Surfaces.
    xImarisApp = getappdata(hSortomatoBase, 'xImarisApp');
    surpassSurfaces = xtgetsporfaces(xImarisApp, 'Surfaces');

    % If the scene has no Surfaces, return.
    if isempty(surpassSurfaces)
        return
    end % if
    
    %% Set the figure and font colors.
    if all(get(hSortomatoBase, 'Color') == [0 0 0])
        bColor = 'k';
        fColor = 'w';

    else
        bColor = 'w';
        fColor = 'k';
        
    end % if
    
    %% Create a GUI to select surfaces.
    sortomatoPos = get(hSortomatoBase, 'Position');
    
    guiWidth = 230;
    guiHeight = 133;
    guiPos = [...
        sortomatoPos(1) + sortomatoPos(3)/2 - guiWidth/2, ...
        sortomatoPos(2) + sortomatoPos(4) - guiHeight - 25, ...
        guiWidth, ...
        guiHeight];
    
    guiSurfaceIntCalc = figure(...
        'CloseRequestFcn', {@closerequestfcn, hSortomatoBase}, ...
        'Color', bColor, ...
        'MenuBar', 'None', ...
        'Name', 'Surface intensities calculation', ...
        'NumberTitle', 'Off', ...
        'Position', guiPos, ...
        'Resize', 'Off', ...
        'Tag', 'guiSurfaceIntCalc');
    
    % Create the Surface selection popup menus.
    uicontrol(...
        'Background', bColor, ...
        'FontSize', 12, ...
        'Foreground', fColor, ...
        'HorizontalAlign', 'Left', ...
        'Position', [10 86 100 24], ...
        'String', 'Surfaces', ...
        'Style', 'text', ...
        'Tag', 'textSurfaces');
    
    popupSurfaces = uicontrol(...
        'Background', bColor, ...
        'FontSize', 12, ...
        'Foreground', fColor, ...
        'Parent', guiSurfaceIntCalc, ...
        'Position', [120 90 100 24], ...
        'String', {surpassSurfaces.Name}, ...
        'Style', 'popupmenu', ...
        'Tag', 'popupSurfaces', ...
        'TooltipString', 'Select surfaces for surface intensity calculation', ...
        'Value', 1);
    
    % Create the calculate button.    
    uicontrol(...
        'Background', bColor, ...
        'Callback', @(pushCalc, eventData)pushcalc(pushCalc, eventData), ...
        'FontSize', 12, ...
        'Foreground', fColor, ...
        'Parent', guiSurfaceIntCalc, ...
        'Position', [130 40 90 24], ...
        'String', 'Calculate', ...
        'Style', 'pushbutton', ...
        'Tag', 'pushCalc', ...
        'TooltipString', 'Calculate surface intensities');
    
    %% Setup the status bar.
    hStatus = statusbar(guiSurfaceIntCalc, '');
    hStatus.CornerGrip.setVisible(false)
    
    hStatus.ProgressBar.setForeground(java.awt.Color.black)
    hStatus.ProgressBar.setString('')
    hStatus.ProgressBar.setStringPainted(true)
    
    %% Add the GUI to the base's GUI children.
    guiChildren = getappdata(hSortomatoBase, 'guiChildren');
    guiChildren = [guiChildren; guiSurfaceIntCalc];
    setappdata(hSortomatoBase, 'guiChildren', guiChildren)
    
    %% Nested function to perform surface intensity calculation
    function pushcalc(varargin)
        % Perform surface intensity calculation
        %
        %
        
        %% Get the Surfaces object.
        calcObjectIdx = get(popupSurfaces, 'Value');
        xObject = xImarisApp.GetFactory.ToSurfaces(...
            surpassSurfaces(calcObjectIdx).ImarisObject);

        %% Get the data set geometry.
        xMin = xImarisApp.GetDataSet.GetExtendMinX;
        yMin = xImarisApp.GetDataSet.GetExtendMinY;
        zMin = xImarisApp.GetDataSet.GetExtendMinZ;

        xMax = xImarisApp.GetDataSet.GetExtendMaxX;
        yMax = xImarisApp.GetDataSet.GetExtendMaxY;
        zMax = xImarisApp.GetDataSet.GetExtendMaxZ;

        % Should shift the dataset to the range 0:max?

        % Calculate the dataset size.
        xSize = xImarisApp.GetDataSet.GetSizeX;
        ySize = xImarisApp.GetDataSet.GetSizeY;
        zSize = xImarisApp.GetDataSet.GetSizeZ;
        
        % Get the number of channels.
        xChannelCount = xImarisApp.GetDataSet.GetSizeC;
        
        %% Get the number of surfaces and allocate arrays for the data.
        calcSurfaceCount = xObject.GetNumberOfSurfaces;
        calcSurfaceTimes = zeros(calcSurfaceCount, 1);
        interiorIntensity = zeros(calcSurfaceCount, xChannelCount);
        surfaceIntensity = zeros(calcSurfaceCount, xChannelCount);
        
        %% Setup the status and progress bars.
        hStatus.setText('Calculating surface intensities');
        hStatus.ProgressBar.setMaximum(calcSurfaceCount)
        hStatus.ProgressBar.setVisible(true)
        
        %% Calculate the internal and surface associated intensity fractions.
        for s = 1:calcSurfaceCount
            % Get the surface time point.
            calcSurfaceTimes(s) = xObject.GetTimeIndex(s - 1);
            
            % Get the surface mask.
            sMask = xObject.GetSingleMask(s - 1, ...
                xMin, yMin, zMin, xMax, yMax, zMax, xSize, ySize, zSize);
            
            % Get the mask data and reshape. The reshape is needed for bwperim.
            mMask = sMask.GetDataVolumeAs1DArrayBytes(0, 0);
            mMask = reshape(mMask, [xSize, ySize, zSize]);
            
            % Find the mask foreground pixel indices and convert to subscripts.
            maskLinearIdxs = find(mMask);
            [maskXs, maskYs, maskZs] = ind2sub([xSize, ySize, zSize], maskLinearIdxs);
            
            % Calculate the bounding box for the mask. These are 1-based
            % indices.
            maskXBounds = [min(maskXs), max(maskXs)];
            maskYBounds = [min(maskYs), max(maskYs)];
            maskZBounds = [min(maskZs), max(maskZs)];

            % Crop the mask to the bounding box.
            subMask = mMask(maskXBounds(1):maskXBounds(2), ...
                maskYBounds(1):maskYBounds(2), maskZBounds(1):maskZBounds(2));
            
            % Get the perimeter mask.
            subShell = bwperim(subMask);
            
            % Need to do something about objects at the edge. Clip the 0/max xyz
            % voxels? Exclude the whole object?
            
            % Calculate the intensities in the surface regions for all channels.
            for c = 1:xChannelCount
                % Get the intensity data. Convert the starting indices to 0-based
                % values.
                subImage = xImarisApp.GetDataSet.GetDataSubVolumeAs1DArrayFloats(...
                    maskXBounds(1) - 1, maskYBounds(1) - 1, maskZBounds(1) - 1, ...
                    c - 1, calcSurfaceTimes(s), ...
                    diff(maskXBounds) + 1, diff(maskYBounds) + 1, diff(maskZBounds) + 1);

                % Calculate the intensity in the interior and on the surface.
                surfaceIntensity(s, c) = sum(subImage(subShell));
                interiorIntensity(s, c) = sum(subImage(~subShell));
            end % for c

            % Delete the reference to the mask and call the garbage collector.
            clear sMask
            java.lang.System.gc
            
            % Update the progress bar.
            hStatus.ProgressBar.setValue(s)
        end % for s
        
        %% Send the interior intensity statistics to Imaris.
        % Update the progress bar.
        hStatus.ProgressBar.setValue(0)
        hStatus.ProgressBar.setMaximum(2)

        % Create the ID list.
        calcSurfaceIDs = repmat(transpose(0:calcSurfaceCount - 1), ...
            [xChannelCount 1]);

        % Calculate the number of statistic entries.
        statCount = xChannelCount*calcSurfaceCount;

        % Create the interior intensity stat name list.
        interiorNames = repmat({'Interior intensity'}, [statCount 1]);

        % Create the unit list.
        statUnits = repmat({''}, [statCount 1]); 

        % Assemble the factors cell array.
        statFactors = cell(4, statCount);

        % Set the Category to Surfaces.
        statFactors(1, :) = repmat({'Surface'}, [statCount 1]);

        % Set the Channels.
        channelNos = num2cell(1:xChannelCount);
        channelStrs = cellfun(@num2str, channelNos, 'UniformOutput', 0);
        channelStatList = repmat(channelStrs, [calcSurfaceCount 1]);
        statFactors(2, :) = channelStatList(:);

        % Set the Collection to an empty string.
        statFactors(3, :) = repmat({''}, [statCount 1]);

        % Replicate the surface time indices for each channel stat.
        calcTimeList = repmat(calcSurfaceTimes + 1, [1 xChannelCount]); 
        
        % Set the Time.
        statFactors(4, :) = num2cell(calcTimeList(:));

        % Convert the time points to strings...
        statFactors(4, :) = cellfun(@num2str, statFactors(4, :), ...
            'UniformOutput', 0);

        % Create the factor names.
        factorNames = {'Category'; 'Channel'; 'Collection'; 'Time'};

        % Send the interior intensities to Imaris.
        xObject.AddStatistics(interiorNames, interiorIntensity(:), statUnits, ...
            statFactors, factorNames, calcSurfaceIDs)

        % Update the progress bar.
        hStatus.ProgressBar.setValue(1)

        %% Transfer the surface intensity statistics to Imaris.
        % Create the surface intensity stat name list.
        surfaceNames = repmat({'Surface intensity'}, [statCount 1]);

        % Send the surface intensities to Imaris.
        xObject.AddStatistics(surfaceNames, surfaceIntensity(:), statUnits, ...
            statFactors, factorNames, calcSurfaceIDs)
        
        hStatus.ProgressBar.setValue(2)
        
        %% Reset the status bar.
        hStatus.setText('')
        hStatus.ProgressBar.setValue(0)
        hStatus.ProgressBar.setVisible(false)
    end % pushcalc    
end % sortomatosurfaceintensity


function closerequestfcn(hObject, ~, hSortomatoBase)
    % Close the sortomato sub-GUI figure
    %
    %
    
    %% Remove the GUI's handle from the base's appdata and delete.
    guiChildren = getappdata(hSortomatoBase, 'guiChildren');

    guiIdx = guiChildren == hObject;
    guiChildren = guiChildren(~guiIdx);
    setappdata(hSortomatoBase, 'guiChildren', guiChildren)
    delete(hObject);
end % closerequestfcn