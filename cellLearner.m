function [  ] = cellLearner(  )
%change to folder above
cd(fileparts(mfilename('fullpath')))

[file, path] = uigetfile('*.mat','Select .mat data file');
if (file == 0)
    return; %canceled
end
dataFile = matfile(strcat(path,file),'Writable',true);

stats = dataFile.stats;
xyzPositions = dataFile.stitchedXYZPositions;
timeIndex = dataFile.designMatrixTimeIndices;  

%createLabels for cells of interest
if (~any(strcmp('coiIndices',who(dataFile))))
    dataFile.coiIndices = [];
    dataFile.ncoiIndices = [];
end
%pull nxp feature matrix and imaris indices in memory
if any(strcmp('features',who(dataFile)))
    features = dataFile.features;
else
   features = dataFile.rawFeatures; 
end
imarisIndices = dataFile.imarisIndices;

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
%print instructions
printManualSelectionInstructions();
printAutomatedSelectionInstructions();

    %Get indices of surfaces sorted by distance to point specified
    function [indices] = getsurfacesnearpoint(point)       
        %Get indices of surfaces in current TP 
        currentTPIndices = find(xImarisApp.GetVisibleIndexT == dataFile.designMatrixTimeIndices); 
        surfaceCentersCell = mat2cell(xyzPositions(currentTPIndices,:),ones(length(currentTPIndices),1),3);
        distances = cellfun(@(surfCenter) sum((surfCenter - point).^2) ,surfaceCentersCell);
        [~, closestIndices] = sort(distances,1,'ascend');
        %get indices corresponding to set of all surfaces, sorted by
        %distance to axis
        %don't take more than 1000
        indices = currentTPIndices(closestIndices(1:1000));        
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
        func_addsurfacestosurpass(xImarisApp,dataFile,1,xSurfaceToClassify,...
            find(imarisIndices(index) == stats(1).Ids));
    end

    function [] = printManualSelectionInstructions()
       fprintf('Manual selection mode: \n');
       fprintf('1: position first 2 crosshairs \n');
       fprintf('2: position 3rd crosshair \n');
       fprintf('3: position 3rd crosshair \n');
       fprintf('4: Yes, this is a cell of interest \n');
       fprintf('5: No, this is not a cell of interest \n');
       fprintf('6: Step backwards through preview \n');
       fprintf('7: Step forwards through preview \n');        
    end

    function [] = printAutomatedSelectionInstructions()
        fprintf('Interactive Learning selection mode: \n');
        fprintf('q: label next example at current timepoint \n');
        fprintf('y: yes, the currently presented example is a cell of interest \n');
        fprintf('n: no, the currently presented example is not a cell of interest \n');
        fprintf('w: classify and visualize all surfaces at current TP \n');
        fprintf('e: classify and visualize all surfaces at all timepoints \n');        
    end
  
%Controls
    function [] = keyinput(~,~)
        key = get(gcf,'CurrentCharacter');
        
                    
        if strcmp(key,'1') %position crosshairs orthagonal to view
            xClip1.SetVisible(true);
            xClip2.SetVisible(true);
            positioncrosshairs();
            printManualSelectionInstructions();
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
             closestSurfaceIndices = getsurfacesnearpoint(point); %these are sorted design matrix indices
             manualPreviewIndex = 1; %start with closest             
             updatepreviewsurface( );
             %hide crosshairs
             xClip1.SetVisible(false);
             xClip2.SetVisible(false);
             %show preview
             xSurfaceToClassify.SetVisible(true);
        elseif strcmp(key,'4') %add previewed surface to persistent set
            %Select in file (so saves as you go)
            indices = unique([dataFile.coiIndices; closestSurfaceIndices(manualPreviewIndex)]);
            dataFile.coiIndices = indices;
            fprintf('total selected cells of interest: %i\n',length(indices));            
        elseif strcmp(key,'5') %previewd is not a cell of interest
            indices = unique([dataFile.ncoiIndices; closestSurfaceIndices(manualPreviewIndex)]);
            dataFile.ncoiIndices = indices;
            fprintf('total selected not cells of interest: %i\n',length(indices));    
        elseif strcmp(key,'6') %preview previous surface
            manualPreviewIndex = max(1,manualPreviewIndex - 1);
            updatepreviewsurface();
        elseif strcmp(key,'7') %preview next surface
            manualPreviewIndex = min(length(closestSurfaceIndices),manualPreviewIndex+1);
            updatepreviewsurface();   
        %%%%%%%%% Active Learning %%%%%%%%%
        elseif strcmp(key,'q')
            if (isempty(dataFile.coiIndices) || isempty(dataFile.ncoiIndices))
               error('Must manually select 2 small populations of cells to train classifier'); 
            end
            printAutomatedSelectionInstructions();
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
            dataFile.coiIndices = unique([dataFile.coiIndices; surfaceClassicationIndex_]);
            classifier = retrain(3);
            presentNextExample();
        elseif strcmp(key,'n')
            %Yes the currently presented instance show be laballed as a T cell
            dataFile.ncoiIndices = unique([dataFile.ncoiIndices; surfaceClassicationIndex_]);
            fprintf('Total cells: %i\nTotal not cells: %i',length(dataFile.coiIndices),length(dataFile.ncoiIndices)); 
            classifier = retrain(3);
            presentNextExample();
        end
    end

    function [cls] = retrain(numLearners)
        %train classifier
        cls = trainClassifier([features(dataFile.coiIndices,:); features(dataFile.ncoiIndices,:)],...
            [ones(length(dataFile.coiIndices),1); zeros(length(dataFile.ncoiIndices),1)],numLearners);
    end

    function [] = predictAll()
        classifier = retrain(50);
        %delete all predeicted surfaces from imaris
        xPopulationSurface.RemoveAllSurfaces;
        %run classifier on instances at current TP
        pred = classify( classifier, features, ones(size(timeIndex),'logical'),...
            dataFile.coiIndices, dataFile.ncoiIndices);
        
        func_addsurfacestosurpass(xImarisApp,dataFile,100, xPopulationSurface,imarisIndices(logical(pred))+1);
  
    end

    function [] = predictCurrentTP()
        classifier = retrain(20);
        %delete all predeicted surfaces from imaris
        xPopulationSurface.RemoveAllSurfaces;
        %run classifier on instances at current TP
        currentTPMask = xImarisApp.GetVisibleIndexT == timeIndex;
        currentTPPred = classify( classifier, features, currentTPMask, dataFile.coiIndices, dataFile.ncoiIndices);
        
        %generate mask for predicted cells of interest at current TP
        currentTPIndices = find(currentTPMask);
        coiAtCurrentTPIndices = currentTPIndices(currentTPPred);
        %send to Imaris
        func_addsurfacestosurpass(xImarisApp,dataFile,100, xPopulationSurface,imarisIndices(coiAtCurrentTPIndices)+1);
    end

    function [] = presentNextExample()
        %find most informative example at this time point (that arent
        %already labelled)
        unlabelledAtCurrentTP = xImarisApp.GetVisibleIndexT == timeIndex;
        unlabelledAtCurrentTP([dataFile.coiIndices; dataFile.ncoiIndices]) = 0;
        [~, currentTPPredValue] = classify( classifier, features, unlabelledAtCurrentTP, dataFile.coiIndices, dataFile.ncoiIndices);
        
        surfaceClassicationIndex_ = nextSampleToClassify( currentTPPredValue, unlabelledAtCurrentTP );
        %remove exisiting surfaces to classify
        xSurfaceToClassify.RemoveAllSurfaces;
        func_addsurfacestosurpass(xImarisApp,dataFile,1, xSurfaceToClassify,imarisIndices(surfaceClassicationIndex_));
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

