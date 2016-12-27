function sortomatoturningangles(~, ~, hSortomatoBase)
    % SORTOMATOTURNINGANGLES Calculate track turning angles
    %   Detailed explanation goes here
    %
    %  ©2010-2013, P. Beemiller. Licensed under a Creative Commmons Attribution
    %  license. Please see: http://creativecommons.org/licenses/by/3.0/
    %
    
    %% Check for an already-running GUI.
    guiChildren = getappdata(hSortomatoBase, 'guiChildren');
    
    if ~isempty(guiChildren)
        guiTurningAngle = findobj(guiChildren, 'Tag', 'guiTurningAngle');
        
        if ~isempty(guiTurningAngle)
            figure(guiTurningAngle)
            return
        end % if
    end % if
    
    %% Get the Surpass Spots and Surfaces.
    xImarisApp = getappdata(hSortomatoBase, 'xImarisApp');
    surpassObjects = xtgetsporfaces(xImarisApp, 'Both');

    % If the scene has no Spots or Surfaces, return.
    if isempty(surpassObjects)
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
    
    %% Create a GUI to select objects.
    sortomatoPos = get(hSortomatoBase, 'Position');
    
    guiWidth = 230;
    guiHeight = 133;
    guiPos = [...
        sortomatoPos(1) + sortomatoPos(3)/2 - guiWidth/2, ...
        sortomatoPos(2) + sortomatoPos(4) - guiHeight - 25, ...
        guiWidth, ...
        guiHeight];
    
    guiTurningAngle = figure(...
        'CloseRequestFcn', {@closerequestfcn, hSortomatoBase}, ...
        'Color', bColor, ...
        'MenuBar', 'None', ...
        'Name', 'Turning angles calculation', ...
        'NumberTitle', 'Off', ...
        'Position', guiPos, ...
        'Resize', 'Off', ...
        'Tag', 'guiTurningAngle');
    
    % Create the object selection popup menu.
    uicontrol(...
        'Background', bColor, ...
        'FontSize', 12, ...
        'Foreground', fColor, ...
        'HorizontalAlign', 'Left', ...
        'Position', [10 86 108 24], ...
        'String', 'Objects', ...
        'Style', 'text', ...
        'Tag', 'textObjects');
    
    popupObjects = uicontrol(...
        'Background', bColor, ...
        'FontSize', 12, ...
        'Foreground', fColor, ...
        'Parent', guiTurningAngle, ...
        'Position', [120 90 100 24], ...
        'String', {surpassObjects.Name}, ...
        'Style', 'popupmenu', ...
        'Tag', 'popupObjects', ...
        'TooltipString', 'Select objects for turning angle calculation', ...
        'Value', 1);
    
    % Create the calculate button.
    uicontrol(...
        'Background', bColor, ...
        'Callback', {@pushcalc}, ...
        'FontSize', 12, ...
        'Foreground', fColor, ...
        'Parent', guiTurningAngle, ...
        'Position', [130 40 90 24], ...
        'String', 'Calculate', ...
        'Style', 'pushbutton', ...
        'Tag', 'pushCalc', ...
        'TooltipString', 'Calculate turning angles');
    
    %% Setup the status bar.
    hStatus = statusbar(guiTurningAngle, '');
    hStatus.CornerGrip.setVisible(false)
    
    hStatus.ProgressBar.setForeground(java.awt.Color.black)
    hStatus.ProgressBar.setString('')
    hStatus.ProgressBar.setStringPainted(true)
    
    %% Add the GUI to the base's GUI children.
    guiChildren = getappdata(hSortomatoBase, 'guiChildren');
    guiChildren = [guiChildren; guiTurningAngle];
    setappdata(hSortomatoBase, 'guiChildren', guiChildren)
    
    %% Nested function to perform turning angle calculation
    function pushcalc(varargin)
        % Calculate turning angles
        %
        %

        %% Get the seleted Surpass object.
        angleObjectIdx = get(popupObjects, 'Value');
        xObject = surpassObjects(angleObjectIdx).ImarisObject;
        
        %% Setup the status bar.
        hStatus.setText('Calculating turning angles')

        %% Get the Surpass object data.
        if xImarisApp.GetFactory.IsSpots(xObject)
            % Get the spot positions.
            objectPos = xObject.GetPositionsXYZ;

            % Get the spot times.
            objectTimes = xObject.GetIndicesT;

        else
            % Get the number of surfaces.
            surfaceCount = xObject.GetNumberOfSurfaces;

            % Get the surface positions and times.
            objectPos = zeros(surfaceCount, 3);
            objectTimes = zeros(surfaceCount, 1);
            for s = 1:surfaceCount
                objectPos(s, :) = xObject.GetCenterOfMass(s - 1);
                objectTimes(s) = xObject.GetTimeIndex(s - 1);
            end % s

        end % if

        %% Create a list of object indexes and get the track information.
        % Create the spot indexes (just a 0-based index).
        objectIdxs = transpose(0:size(objectPos, 1) - 1);

        % Get the track information.
        objectIDs = xObject.GetTrackIds;
        objectEdges = xObject.GetTrackEdges;
        trackLabels = unique(objectIDs);

        %% Calculate the turning angles.
        % Allocate a vector for the turning angle data.
        objectAngles = zeros(size(objectIdxs));

        % Allocate a vector for the track turning angle data.
        trackAngles = zeros(size(trackLabels));

        % Setup the progress bar.
        hStatus.ProgressBar.setValue(0)
        hStatus.ProgressBar.setMaximum(length(trackLabels))
        hStatus.ProgressBar.setVisible(true)

        % Calculate the turning angle for all the tracks.
        for r = 1:length(trackLabels)
            % Get indices for the track.
            rEdges = objectEdges(objectIDs == trackLabels(r), :);
            rObjects = unique(rEdges);

            % Get the track positions.
            rPos = objectPos(rObjects + 1, :);

            % Calculate the track movement vectors.
            rVectors = diff(rPos);

            % Calculate the turning angles:

            % Get the vectors.
            inVectors = rVectors(1:end - 1, :);
            outVectors = rVectors(2:end, :);

            % Calculate the cross products.
            vectorCrosses = cross(inVectors, outVectors, 2);

            % Calculate the magnitudes.
            vectorMags = sqrt(sum(vectorCrosses.^2, 2));

            % Calculate the dot procuts.
            vectorDots = dot(inVectors, outVectors, 2);

            % Calculate the angles in degrees.
            rAngles = atand(vectorMags./vectorDots);

            % Remap the angles from the range [-pi/2, pi/2] to [0, pi].
            rAngles(rAngles < 0) = 180 + rAngles(rAngles < 0);

            % Add the angles to the list. Convert the Imaris spot indices to
            % 1-based indices and insert the turning angles starting at the
            % 2nd position in the track (there is no turning angle for t0 or
            % tfinal).
            objectAngles(rObjects(2:end - 1) + 1) = rAngles;

            % Calculate the track average turning angle.
            trackAngles(r) = mean(rAngles);

            % Update the progress bar.
            hStatus.ProgressBar.setValue(r)
        end % for r

        %% Transfer the turning angle stats to Imaris.
        % Update the status bar and progress bar.
        hStatus.setText('Transferring turning angle data')
        hStatus.ProgressBar.setValue(0)
        hStatus.ProgressBar.setMaximum(2)

        % Create the stat name list.
        angleNames = repmat({'Turning angle'}, size(objectIdxs));

        % Create the unit list.
        imarisUnits = 'Degrees';
        angleUnits = repmat({imarisUnits}, size(objectIdxs)); 

        % Assemble the factors cell array.
        angleFactors = cell(3, length(objectIdxs));

        % Set the Category.
        if xImarisApp.GetFactory.IsSpots(xObject)
            angleFactors(1, :) = repmat({'Spot'}, size(objectIdxs));

        else
            angleFactors(1, :) = repmat({'Surface'}, size(objectIdxs));

        end

        % Set the Collection to an empty string.
        angleFactors(2, :) = repmat({''}, size(objectIdxs));
        
        % Set the time.
        angleFactors(3, :) = num2cell(objectTimes + 1);

        % Convert the time points to strings...
        angleFactors(3, :) = cellfun(@num2str, angleFactors(3, :), ...
            'UniformOutput', 0);

        % Create the factor names.
        angleFactorNames = {'Category'; 'Collection'; 'Time'};

        % Send the stats to Imaris.
        xObject.AddStatistics(angleNames, objectAngles, angleUnits, ...
            angleFactors, angleFactorNames, objectIdxs)

        % Update the progress bar.
        hStatus.ProgressBar.setValue(1)

        %% Transfer the track turning angle stats to Imaris.
        % Create the stat name list.
        trackAngleNames = repmat({'Track turning angle'}, size(trackLabels));

        % Create the unit list.
        trackAngleUnits = repmat({imarisUnits}, size(trackLabels)); 

        % Assemble the factors cell array.
        trackAngleFactors = cell(3, length(trackLabels));

        % Set the Category to tracks.
        trackAngleFactors(1, :) = repmat({'Track'}, size(trackLabels));

        % Set the Collection to any empty string.
        trackAngleFactors(2, :) = repmat({''}, size(trackLabels));

        % Set the Time to an empty string.
        trackAngleFactors(3, :) = repmat({''}, size(trackLabels));

        % Send the stats to Imaris.
        xObject.AddStatistics(trackAngleNames, trackAngles, trackAngleUnits, ...
            trackAngleFactors, angleFactorNames, trackLabels)
        
        % Update the progress bar.
        hStatus.ProgressBar.setValue(2)
        
        %% Reset the progress and status bars.
        hStatus.setText('')
        hStatus.ProgressBar.setValue(0)
        hStatus.ProgressBar.setVisible(false)
    end % pushcalc    
end % sortomatoturningangles


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