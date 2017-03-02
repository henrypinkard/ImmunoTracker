function [  ] = cellPopulationLearner(  )
%ACTIVELEARNINGCLASSIFIER Master function for querying user and classifying surfaces
% This function loads an unlabelled nxp data matrix and prompts the user
% for labeling of examples so that it can learn and generalize to the full
% dataset as fast as possible
%
%
% CONTROLS:
% 1 - Enter active learning mode: begin presenting unlabelled examples at
% current time point
% 2 - Classify and visualize all instances at current time point
% 3 - Activate crosshair selection mode to manually select an instance to
% classify (needed if for example, the active learning struggles with a
% particular cell
% y - Yes the currently presented instance show be laballed as a T cell
% n - No the currently presented instance in not a T cell

%start parallel pool for training NN
% parpool;

%change to folder above
cd(fileparts(mfilename('fullpath')))
cd('..')

dataFile = '/Users/henrypinkard/Google Drive/Research/BIDC/LNImagingProject/data/CMTMRFeaturesAndLabels.mat';
surfaceFile = '/Users/henrypinkard/Google Drive/Research/BIDC/LNImagingProject/data/CMTMRCandidates.mat';
%load features and known labels
featuresAndLabels = matfile(dataFile,'Writable',true);
%Load surface data into virtual memory
surfaceData = matfile(surfaceFile,'Writable',false);
%cache stats
stats = surfaceData.stats;
xyzPositions = featuresAndLabels.stitchedXYZPositions;


%createLabels for cells of interest
if (~any(strcmp('coiIndices',who(featuresAndLabels))))
    featuresAndLabels.coiIndices = [];
    featuresAndLabels.ncoiIndices = [];
end
%pull nxp feature matrix and imaris indices in memory
features = featuresAndLabels.features;
imarisIndices = featuresAndLabels.imarisIndices;

%Connect to Imaris
[ xImarisApp, xPopulationSurface, xSurfaceToClassify ] = xtSetupSurfaceTransfer(  );
xSurpassCam = xImarisApp.GetSurpassCamera;
%make stuff for manual selection
%delete old  clipping planes
for k = xSurpass.GetNumberOfChildren - 1 :-1: 0
   if  strcmp(char(xSurpass.GetChild(k).GetName),'Clip 1') || strcmp(char(xSurpass.GetChild(k).GetName),'Clip 2') 
      xSurpass.RemoveChild(xSurpass.GetChild(k)); 
   end
end
xClip1 = xImarisApp.GetFactory.CreateClippingPlane;
xClip2 = xImarisApp.GetFactory.CreateClippingPlane;
xClip1.SetName('Clip 1');
xClip2.SetName('Clip 2');
xClip1.SetVisible(false);
xClip2.SetVisible(false);
xSurpass.AddChild(xClip1,-1);
xSurpass.AddChild(xClip2,-1);
closestSurfaceIndices = [];
manualPreviewIndex = 1; 
%create holder for first two planes from crosshairs
planeMatrix = zeros(3,3);
planeConstants = zeros(3,1);
%make functions for manipulating view
[quatMult,quatRot,rotVect2Q] = makeQuaterionFunctions();



%create figure for key listening
figure(1);
title('Imaris bridge');
set(gcf,'KeyPressFcn',@keyinput);
surfaceClassicationIndex_ = -1;

%train initial classifier
classifier = retrain(3);



    %Get indices of surfaces sorted by distance to point specified
    function [indices] = getsurfacesnearpoint(point)       
        %Get indices of surfaces in current TP 
        currentTPIndices = find(xImarisApp.GetVisibleIndexT == featuresAndLabels.timeIndices); 
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

    function [] = updatepreviewsurface()
        %instance into deisgn matrix
        index = closestSurfaceIndices(manualPreviewIndex);
        xSurfaceToClassify.RemoveAllSurfaces; %clear old ones
        func_addsurfacestosurpass(xImarisApp,surfaceData,1,xSurfaceToClassify,...
            find(featuresAndLabels.imarisIndices(index,1) == stats(1).Ids));
    end

%Controls
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
             manualPreviewIndex = 1; %start with closest
             updatepreviewsurface( );
             %hide crosshairs
             xClip1.SetVisible(false);
             xClip2.SetVisible(false);
             %show preview
             xSurfaceToClassify.SetVisible(true);
        elseif strcmp(key,'4') %add previewed surface to persistent set
            %Select in file (so saves as you go)
            indices = unique([featuresAndLabels.coiIndices; closestSurfaceIndices(manualPreviewIndex)]);
            featuresAndLabels.coiIndices = indices;
            fprintf('total selected cells of interest: %i\n',length(indices));            
        elseif strcmp(key,'5') %previewd is not a cell of interest
            indices = unique([featuresAndLabels.ncoiIndices; closestSurfaceIndices(manualPreviewIndex)]);
            featuresAndLabels.ncoiIndices = indices;
            fprintf('total selected not cells of interest: %i\n',length(indices));    
        elseif strcmp(key,'6') %preview previous surface
            manualPreviewIndex = max(1,manualPreviewIndex - 1);
            updatepreviewsurface();
        elseif strcmp(key,'7') %preview next surface
            manualPreviewIndex = min(length(closestSurfaceIndices),manualPreviewIndex+1);
            updatepreviewsurface();   
        %%%%%%%%% Active Learning %%%%%%%%%
        elseif strcmp(key,'q')
            if (isempty(featuresAndLabels.coiIndices) || isempty(featuresAndLabels.ncoiIndices))
               error('Must manually select 2 small populations of cells to train classifier'); 
            end
            % Enter active learning mode: begin presenting unlabelled examples at current time point            
            classifier = retrain(3);
            presentNextExample();
        elseif strcmp(key,'w')
            % Classify and visualize all instances at current time point
            fprintf('Classifying all surfaces at current time point...\n');
            predictCurrentTP();
        elseif strcmp(key,'e')
            % Classify all
            fprintf('Classifying all surfaces...\n');
            predictAll();
            %TODO: prompt for filename and export
        elseif strcmp(key,'y')
            %Yes the currently presented instance show be laballed as a T cell
            featuresAndLabels.coiIndices = unique([featuresAndLabels.coiIndices; surfaceClassicationIndex_]);
            classifier = retrain(3);
            presentNextExample();
        elseif strcmp(key,'n')
            %Yes the currently presented instance show be laballed as a T cell
            featuresAndLabels.ncoiIndices = unique([featuresAndLabels.ncoiIndices; surfaceClassicationIndex_]);
            classifier = retrain(3);
            presentNextExample();
        end
    end

    function [cls] = retrain(numLearners)
        %train classifier
        cls = trainClassifier([features(featuresAndLabels.coiIndices,:); features(featuresAndLabels.ncoiIndices,:)],...
            [ones(length(featuresAndLabels.coiIndices),1); zeros(length(featuresAndLabels.ncoiIndices),1)],numLearners);
    end

    function [] = predictAll()
        classifier = retrain(50);
        %delete all predeicted surfaces from imaris
        xPopulationSurface.RemoveAllSurfaces;
        %run classifier on instances at current TP
        pred = classify( classifier, features, ones(size(featuresAndLabels.timeIndices),'logical'),...
            featuresAndLabels.coiIndices, featuresAndLabels.ncoiIndices);
        
        func_addsurfacestosurpass(xImarisApp,surfaceData,100, xPopulationSurface,imarisIndices(logical(pred)));
  
    end

    function [] = predictCurrentTP()
        classifier = retrain(20);
        %delete all predeicted surfaces from imaris
        xPopulationSurface.RemoveAllSurfaces;
        %run classifier on instances at current TP
        currentTPMask = xImarisApp.GetVisibleIndexT == featuresAndLabels.timeIndices;
        currentTPPred = classify( classifier, features, currentTPMask, featuresAndLabels.coiIndices, featuresAndLabels.ncoiIndices);
        
        %generate mask for predicted cells of interest at current TP
        currentTPIndices = find(currentTPMask);
        coiAtCurrentTPIndices = currentTPIndices(currentTPPred);
        %send to Imaris
        func_addsurfacestosurpass(xImarisApp,surfaceData,100, xPopulationSurface,imarisIndices(coiAtCurrentTPIndices));
    end

    function [] = presentNextExample()
        %find most informative example at this time point (that arent
        %already labelled)
        unlabelledAtCurrentTP = xImarisApp.GetVisibleIndexT == featuresAndLabels.timeIndices;
        unlabelledAtCurrentTP([featuresAndLabels.coiIndices; featuresAndLabels.ncoiIndices]) = 0;
        [~, currentTPPredValue] = classify( classifier, features, unlabelledAtCurrentTP, featuresAndLabels.coiIndices, featuresAndLabels.ncoiIndices);
        
        surfaceClassicationIndex_ = nextSampleToClassify( currentTPPredValue, unlabelledAtCurrentTP );
        %remove exisiting surfaces to classify
        xSurfaceToClassify.RemoveAllSurfaces;
        func_addsurfacestosurpass(xImarisApp,surfaceData,1, xSurfaceToClassify,imarisIndices(surfaceClassicationIndex_));
        centerToSurface(xSurfaceToClassify);
    end

    function [] = centerToSurface(surface)
        %surpass camera parameters:
        %height--distance from the object it is centered on, changes with zoom but not with rotation
        %position--camera position, not related to center of rotation
        %focus--seemingly not related to anything... (maybe doesn't matter for orthgraphic)
        %Fit will fit viewing frame around entire image...which will in turn set the center of rotation to the center of image
        
        %get initial height for resetting
        height = xSurpassCam.GetHeight;
        %Oreint top down
        xSurpassCam.SetOrientationQuaternion(quatRot(pi,[1; 0; 0])); %top down view
        %Make center of rotation center of entire image
        xSurpassCam.Fit;
        currentCamPos = xSurpassCam.GetPosition;
        previewPos = surface.GetCenterOfMass(0);
        xSurpassCam.SetPosition([previewPos(1:2), currentCamPos(3) ]); %center camera to position of preview surface
        xSurpassCam.SetHeight(height);
        %make sure preview is visible
        surface.SetVisible(true);
    end

    function [ xImarisApp, xPopulationSurface, xSurfaceToClassify ] = xtSetupSurfaceTransfer(  )
        previewName = 'Preview surface';
        populationName = 'Cells of interest';
        surfaceToClassifyName ='SurfaceToClassify';
        
        xtIndex = 0;
        javaaddpath('./ImarisLib.jar')
        vImarisLib = ImarisLib;
        xImarisApp = vImarisLib.GetApplication(xtIndex);
        if (isempty(xImarisApp))
            error('Wrong imaris index');
        end
        
        xSurpass = xImarisApp.GetSurpassScene;
        %delete old surface preview
        for i = xSurpass.GetNumberOfChildren - 1 :-1: 0
            if (strcmp(char(xSurpass.GetChild(i).GetName),previewName) || strcmp(char(xSurpass.GetChild(i).GetName),populationName)...
                    || strcmp(char(xSurpass.GetChild(i).GetName),surfaceToClassifyName))
                xSurpass.RemoveChild(xSurpass.GetChild(i));
            end
        end
        xPopulationSurface = xImarisApp.GetFactory.CreateSurfaces;
        xPopulationSurface.SetName(populationName);
        xSurpass.AddChild(xPopulationSurface,-1);
        xSurfaceToClassify = xImarisApp.GetFactory.CreateSurfaces;
        xSurfaceToClassify.SetName(surfaceToClassifyName);
        xSurpass.AddChild(xSurfaceToClassify,-1);
        xSurfaceToClassify.SetColorRGBA(16646399); %teal
        xPopulationSurface.SetColorRGBA(16711424); %magenta
    end

    function [quatMult,quatRot,rotVect2Q] = makeQuaterionFunctions()
        % quaternion functions %
        %multiply 2 quaternions
        quatMult = @(q1,q2) [ q1(4).*q2(1:3) + q2(4).*q1(1:3) + cross(q1(1:3),q2(1:3)); q1(4)*q2(4) - dot(q1(1:3),q2(1:3))];
        %generate a rotation quaternion
        quatRot = @(theta,axis) [axis / norm(axis) * sin(theta / 2); cos(theta/2)];
        %Rotate a vector and return a quaternion, which then needs to be subindexed to get a vector again
        rotVect2Q = @(vec,quat) quatMult(quat,quatMult([vec; 0],[-quat(1:3); quat(4)]));
    end

end