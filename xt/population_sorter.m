function [] = population_sorter()
%controls:
%%%%%%%Manual XYZ selection mode%%%%%%%%%%
%1 -- position crosshairs orthagonal to view
%2 -- store planes from positioning of first two crosshiars
%3 -- find point at intersection of three positioned planes and preview closest surface
%5 -- preview previous surface
%6 -- preview next surface
%4 -- add previewed surface to set of objects of interest
%7 -- print imaris indices

%%%%%%Automatic single surface selection mode%%%%%%%%
%q -- show closest by intensity vector distance sort to median of objects of interest (at this time point)
%w -- No, the previewed surface is not of interest
%e -- yes, the previewed surface is a cell of interest

%Tunable parameters
%Which channels are used in calculating spectral profile. Addd in 0s to
%ignore channels
channelsToUse = [1 1 1 1 1 1];
imarisIndex = 0;


javaaddpath ImarisLib.jar

vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(imarisIndex);
if (isempty(xImarisApp))
    msgbox('Wrong imaris index');
    return;
end

% %get data set info
% numTimePoints = xImarisApp.GetDataSet.GetSizeT;

%get file
[filename, pathname] = uigetfile('*.mat');
if (filename == 0)
    return;
end

surfFile = matfile(strcat(pathname,filename),'Writable',true);
stats = surfFile.stats;
%get positions of each surface
xPositions = stats(find(strcmp({stats.Name},'Stitched Position X'))).Values;
yPositions = stats(find(strcmp({stats.Name},'Stitched Position Y'))).Values;
zPositions = stats(find(strcmp({stats.Name},'Stitched Position Z'))).Values;
xyzPositions = [xPositions yPositions zPositions];
timeIndices = stats(find(strcmp({stats.Name},'Time Index'))).Values;

xSurpass = xImarisApp.GetSurpassScene;

%delete old surface preview and clipping planes
for i = xSurpass.GetNumberOfChildren - 1 :-1: 0
   if (strcmp(char(xSurpass.GetChild(i).GetName),'Preview surface') ||...
           strcmp(char(xSurpass.GetChild(i).GetName),'Clip 1') ||...
           strcmp(char(xSurpass.GetChild(i).GetName),'Clip 2') )
      xSurpass.RemoveChild(xSurpass.GetChild(i)); 
   end
end


%create and add selection clipping planes, surface preview, and saved
%surfaces
xClip1 = xImarisApp.GetFactory.CreateClippingPlane;
xClip2 = xImarisApp.GetFactory.CreateClippingPlane;
xClip1.SetName('Clip 1');
xClip2.SetName('Clip 2');
xClip1.SetVisible(false);
xClip2.SetVisible(false);
xPreviewSurface = xImarisApp.GetFactory.CreateSurfaces;
xPreviewSurface.SetName('Preview surface');
xSurfacesOfInterest = xImarisApp.GetFactory.CreateSurfaces;
%xSurfacesMaybe = xImarisApp.GetFactory.CreateSurfaces;
%get pointer to surpass camera
xSurpassCam = xImarisApp.GetSurpassCamera;

%get name of selected indices
selectionName = char(inputdlg('Enter object selection name'));
%get name of maybe indices
% maybeName = char(inputdlg('Enter object maybe name'));
%replace any spaces with underscores
selectionName = regexprep(strtrim(selectionName),'\s+','');
if (any(strcmp(selectionName,who(surfFile))))
    %0 indexed, as in stats in mat file
    selectedImarisIDs = eval(strcat('surfFile.',selectionName)); %read from file
    %add all surfaces on interest in file from previous session
    func_addsurfacestosurpass(xImarisApp,surfFile,1,xSurfacesOfInterest,selectedImarisIDs);         
else
    selectedImarisIDs =[]; %create empty holder;    
end
xSurfacesOfInterest.SetName(selectionName);

%add objects
xSurpass.AddChild(xClip1,-1);
xSurpass.AddChild(xClip2,-1);
xSurpass.AddChild(xPreviewSurface,-1);
xSurpass.AddChild(xSurfacesOfInterest,-1);
%xSurpass.AddChild(xSurfacesMaybe, -1);

%create holder for closest surface indices
closestSurfaceIndices = [];
%create holder for most similar surface indices
mostSimilarSurfaceIndices = [];
%create holder for indices of surfaces that have been rejected
rejectedSurfaceIndices = [];
%create holder for first two planes from crosshairs
planeMatrix = zeros(3,3);
planeConstants = zeros(3,1);
%previewIndex for potential surfaces
previewIndex = 1;


%create figure for key listening
figure(1);
title('Imaris bridge');
set(gcf,'KeyPressFcn',@keyinput);
% set(gcf,'WindowScrollWheelFcn',@mousewheelinput);
set (gcf, 'WindowButtonMotionFcn', @mouseMove);

% quaternion functions %
%multiply 2 quaternions
quatMult = @(q1,q2) [ q1(4).*q2(1:3) + q2(4).*q1(1:3) + cross(q1(1:3),q2(1:3)); q1(4)*q2(4) - dot(q1(1:3),q2(1:3))];
%generate a rotation quaternion
quatRot = @(theta,axis) [axis / norm(axis) * sin(theta / 2); cos(theta/2)];
%Rotate a vector and return a quaternion, which then needs to be subindexed
%to get a vector again
rotVect2Q = @(vec,quat) quatMult(quat,quatMult([vec; 0],[-quat(1:3); quat(4)])); 

    %Get indices of surfaces sorted by distance to point specified
    function [indices] = getsurfacesnearpoint(point)       
        %Get indices of surfaces in current TP 
        currentTPIndices = find(xImarisApp.GetVisibleIndexT == timeIndices); 
        surfaceCentersCell = mat2cell(xyzPositions(currentTPIndices,:),ones(length(currentTPIndices),1),3);
        distances = cellfun(@(surfCenter) sum((surfCenter - point).^2) ,surfaceCentersCell);
        [~, closestIndices] = sort(distances,1,'ascend');
        %get indices corresponding to set of all surfaces, sorted by
        %distance to axis
        indices = currentTPIndices(closestIndices);        
    end

    
    function [intersectionPoint] = getcrosshairintersectionpoint() 
        q1 = xClip1.GetOrientationQuaternion;
        %get normals to clipping planes
        normal = rotVect2Q([0;0;1],q1);
        normal = normal(1:3);
        %get center positions of clipping planes
        pos = xClip1.GetPosition;
        %find point at intersection of previous line
        planeMatrix(3,:) = normal';
        planeConstants(3) = dot(normal,pos);
        intersectionPoint = (planeMatrix \ planeConstants)';
    end

    %Store parameters from two croshair planes
    function  storecrosshairplanes()
        q1 = xClip1.GetOrientationQuaternion;
        q2 = xClip2.GetOrientationQuaternion;
        %get normals to clipping planes
        norm1 = rotVect2Q([0;0;1],q1);
        norm1 = norm1(1:3);
        norm2 = rotVect2Q([0;0;1],q2);
        norm2 = norm2(1:3);
        %get center positions of clipping planes
        pos1 = xClip1.GetPosition;
        pos2 = xClip2.GetPosition;
        %store these two planes 
        planeMatrix(1,:) = norm1';
        planeMatrix(2,:) = norm2';
        planeConstants(1:2) = [dot(norm1,pos1); dot(norm2,pos2)];
            
    end

    %set clipping planes perpendicular to view and eachother so they
    %finction as crosshairs
    function [] = positioncrosshairs()
       %get camera orientation axis, and set clipping planes mutually
       %perpendicular to it and perpendicular to each other
       
       %Quaternions for image (surpass camera) are based on coordinates of
       %the screen, with z axis coming out of the plane of the screen
       %Quaternions for clipping planes are based on the axis of the image
       
       %first three entries of quaternion are vector of axis of rotation,
       %final one is amount of rotation

       %get camera quaternion
       xSurpassCam.SetOrthographic(true);
       camQ = xSurpassCam.GetOrientationQuaternion;
       xClip1.SetOrientationQuaternion(quatMult(camQ,quatRot(pi/2,[1;0;0])));
       xClip2.SetOrientationQuaternion(quatMult(camQ,quatRot(pi/2,[0;1;0])));
    end

    function [] = updatepreviewsurface(id)
        xPreviewSurface.RemoveAllSurfaces; %clear old ones
        func_addsurfacestosurpass(xImarisApp,surfFile,1,xPreviewSurface,stats(1).Ids(id));
    end

    function [distanceSortedMatlabIndices, sortedDistances] = recalcintensityvectordistancesort(useAllTPs)
        %Mask by current TP, ranks surfaces by intensity vector distance,
        %and remove those already added to set
        
        %mask by current time point and other stats
%         spIdx = find(ismember({stats.Name},'Sphericity'));
        nvIdx = find(ismember({stats.Name},'Number of Voxels'));        
        redSumIdx = find(ismember({stats.Name},'Intensity Sum - Channel 5'));
        if (useAllTPs)
            maskedIndices = find(timeIndices > -1);
        else
            %EDIT AS NEEDED
            %mask by current time point
            maskedIndices = find(xImarisApp.GetVisibleIndexT == timeIndices...
                & stats(redSumIdx).Values > 1000);
        end
       
       [norms, distancesToMedian, minDistancesToAny] = intensityvectordistancesort(stats,channelsToUse,selectedImarisIDs, maskedIndices);
               
        distances = distancesToMedian;

        [sortedDistances, closestIndices] = sort(distances,1,'ascend');
        %get indices corresponding to set of all surfaces, sorted by distance to axis
        distanceSortedMatlabIndices = maskedIndices(closestIndices);
    end

    %surpass camera parameters:
    %height--distance from the object it is centered on, changes with zoom but not with rotation
    %position--camera position, not related to center of rotation
    %focus--seemingly not related to anything... (maybe doesn't matter for orthgraphic)
    %Fit will fit viewing frame around entire image...which will in turn set the center of rotation to the center of image
    function [] = centertopreview()
        %get initial height for resetting
        height = xSurpassCam.GetHeight;
        %Oreint top down
        xSurpassCam.SetOrientationQuaternion(quatRot(pi,[1; 0; 0])); %top down view
        %Make center of rotation center of entire image
        xSurpassCam.Fit;
        currentCamPos = xSurpassCam.GetPosition;
        previewPos = xPreviewSurface.GetCenterOfMass(0);
        xSurpassCam.SetPosition([previewPos(1:2), currentCamPos(3) ]); %center camera to position of preview surface    
        xSurpassCam.SetHeight(height);
        %make sure preview is visible
        xPreviewSurface.SetVisible(true);
    end


    function [] = initautofindmode()
        [mostSimilarSurfaceIndices, distances] = recalcintensityvectordistancesort(false);
        previewIndex = 1;
        %figure out which surfaces ave already been marked as rejected or already added to avoid repeating them
        toSkipIndices = find(ismember(stats(1).Ids(mostSimilarSurfaceIndices),[rejectedSurfaceIndices; selectedImarisIDs]));
        %keep skipping ahead until a viable one found
        while (~isempty(find(toSkipIndices == previewIndex, 1)))
            previewIndex = previewIndex + 1;
            if (previewIndex > length(mostSimilarSurfaceIndices))
                msgbox('end of surfaces at timepoint reached');
                break;
            end
        end
        updatepreviewsurface(mostSimilarSurfaceIndices(previewIndex));
        centertopreview();
        
        %plot histogram of data
        bins = linspace(min(distances),max(distances),100);
        hist(distances,bins)
        xlabel('Hypersphere distance')
        ylabel('Count')
        set(findobj(gca,'Type','patch'),'FaceColor','g','EdgeColor','g')
        
        %draw hist showing those that are already excluded
        hold on
        hist(distances(1:length(toSkipIndices)),bins)
        hold off
    end
    
    function mouseMove (object, eventdata)
        set(object,'Visible','on')
%        drawnow
    end

    function [selectedIndicesAtCurrentTP] = getSelectedSurfaceIndicesAtCurrentTP()
        %Get indices of selected surfaces in current TP
        currentTPIndices = find(xImarisApp.GetVisibleIndexT == timeIndices) - 1; 
        selectedIndices = eval(sprintf('surfFile.%s',selectionName));
        selectedIndicesAtCurrentTP = intersect(currentTPIndices,selectedIndices);
    end

    function [] = reviewNextSelectedSurface()
        selectedIndicesAtCurrentTP = getSelectedSurfaceIndicesAtCurrentTP();
        %check if reached end of list
        if previewIndex > length(selectedIndicesAtCurrentTP)
            %move on to next time point, or tell user all done
            if xImarisApp.GetVisibleIndexT == max(timeIndices)
                msgbox('Last time point complete')
                return
            end
            xImarisApp.SetVisibleIndexT(xImarisApp.GetVisibleIndexT + 1)
            %update for new TP
            selectedIndicesAtCurrentTP = getSelectedSurfaceIndicesAtCurrentTP();
        end

        %update preview surface
        xPreviewSurface.RemoveAllSurfaces; %clear old ones
        func_addsurfacestosurpass(xImarisApp,surfFile,1,xPreviewSurface,selectedIndicesAtCurrentTP(previewIndex));
        centertopreview();
        fprintf('Reviewing surface %i of %i at current Time point\n',previewIndex,length(selectedIndicesAtCurrentTP))
    end

    function [] = keyinput(~,~)
        key = get(gcf,'CurrentCharacter');
            
        if strcmp(key,'1') %position crosshairs orthagonal to view
            xClip1.SetVisible(true);
            xClip2.SetVisible(true);
            positioncrosshairs();
        elseif strcmp(key,'2')
            %store planes from positioning of first two crosshiars
            storecrosshairplanes();
            %show 1 set of crosshairs to get another position
            xClip1.SetVisible(true);
            xClip2.SetVisible(false);
            %reposition crosshairs
            positioncrosshairs();
        elseif strcmp(key,'3')  
             %find point at intersection of three positioned planes
             point = getcrosshairintersectionpoint();
             closestSurfaceIndices = getsurfacesnearpoint(point);
             previewIndex = 1; %start with closest
             updatepreviewsurface(closestSurfaceIndices(previewIndex));
             %hide crosshairs
             xClip1.SetVisible(false);
             xClip2.SetVisible(false);
             %show preview
             xPreviewSurface.SetVisible(true);
        elseif strcmp(key,'5') %preview previous surface
           previewIndex = max(1,previewIndex - 1);
            updatepreviewsurface(closestSurfaceIndices(previewIndex));
        elseif strcmp(key,'6') %preview next surface
            previewIndex = min(length(closestSurfaceIndices),previewIndex+1);
            updatepreviewsurface(closestSurfaceIndices(previewIndex));
        elseif strcmp(key,'4') %add previewed surface to persistent set
            selectedImarisIDs = unique([selectedImarisIDs; stats(1).Ids(closestSurfaceIndices(previewIndex))]);
            fprintf('total selected surfaces: %i\n',length(selectedImarisIDs));
            %Select in file (so saves as you go)
            eval(strcat('surfFile.',selectionName,' = selectedImarisIDs;'));
            %actaully add surface to imaris (for viewing purposses only)
            func_addsurfacestosurpass(xImarisApp,surfFile,1,xSurfacesOfInterest,stats(1).Ids(closestSurfaceIndices(previewIndex)));         
         elseif strcmp(key,'7') %print selectedImarisIndices
            selectedImarisIDs
         elseif strcmp(key,'q') %show closest by intensity vector distance sort to median of persistent set (at this time point)
            initautofindmode();
        elseif strcmp(key,'w') %No the previewed surface is not of interest
            %current surface is not a cell of interest so mark it as rejected to avoid reshowing
            rejectedSurfaceIndices = [rejectedSurfaceIndices; stats(1).Ids(mostSimilarSurfaceIndices(previewIndex))];
            %figure out which surfaces ave already been marked as rejected or already added to avoid repeating them
            toSkipIndices = find(ismember(stats(1).Ids(mostSimilarSurfaceIndices),[rejectedSurfaceIndices; selectedImarisIDs]));
            %move to next surface
            previewIndex = previewIndex + 1;
            %keep skipping ahead until a viable one found
            while (~isempty(find(toSkipIndices == previewIndex, 1)))
               previewIndex = previewIndex + 1; 
               if (previewIndex > length(mostSimilarSurfaceIndices))
                   msgbox('end of surfaces at timepoint reached');
                   break;
               end
            end
            updatepreviewsurface(mostSimilarSurfaceIndices(previewIndex));          
            centertopreview();
        elseif strcmp(key,'e') % yes the previewed surface is a cell of interest
            %select in file
            selectedImarisIDs = unique([selectedImarisIDs; stats(1).Ids(mostSimilarSurfaceIndices(previewIndex))]);
            fprintf('total selected surfaces: %i\n',length(selectedImarisIDs));
            %Select in file (so saves as you go)
            eval(strcat('surfFile.',selectionName,' = selectedImarisIDs;'));
            %add surface to imaris (for viewing purposses only)
            func_addsurfacestosurpass(xImarisApp,surfFile,1,xSurfacesOfInterest,stats(1).Ids(mostSimilarSurfaceIndices(previewIndex)));               
            initautofindmode();


        elseif strcmp(key,'z') %review previously selected set
            selectedIndicesAtCurrentTP = getSelectedSurfaceIndicesAtCurrentTP();
            previewIndex = 1;  %start with fisrt selected surface at current time point                   
            %update preview surface
            xPreviewSurface.RemoveAllSurfaces; %clear old ones
            func_addsurfacestosurpass(xImarisApp,surfFile,1,xPreviewSurface,selectedIndicesAtCurrentTP(previewIndex));       
            centertopreview();
            fprintf('Reviewing surface %i of %i at current Time point\n',previewIndex,length(selectedIndicesAtCurrentTP))     
        elseif strcmp(key,'x') %go on to next in previously selected set
            %update preview index
            previewIndex = previewIndex + 1;                   
            reviewNextSelectedSurface();
         elseif strcmp(key,'c') %remove current and go on to next
            selectedIndicesAtCurrentTP = getSelectedSurfaceIndicesAtCurrentTP();
            selectedIndicesInFile = eval(sprintf('surfFile.%s',selectionName));
            temp = selectedIndicesInFile;
            temp(find(selectedIndicesInFile == selectedIndicesAtCurrentTP(previewIndex))) = []; 
            eval(sprintf('surfFile.%s=temp',selectionName));
            %Remove surface from object in imaris
            xSurfacesOfInterest.RemoveSurface(find(selectedIndicesInFile == selectedIndicesAtCurrentTP(previewIndex)) - 1);
            reviewNextSelectedSurface();
        end
        
    end
end




