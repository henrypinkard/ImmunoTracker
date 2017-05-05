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
%createLabels for cells of interest
if (~any(strcmp('coiIndices',who(dataFile))))
    dataFile.coiIndices = [];
    dataFile.ncoiIndices = [];
end
%cache values for performance
if any(strcmp('features',who(dataFile)))
    features = dataFile.features;
else
   features = dataFile.rawFeatures; 
end
imarisIndices = dataFile.imarisIndices;
coiIndices = dataFile.coiIndices;
ncoiIndices = dataFile.ncoiIndices;
designMatrixTimeIndices = dataFile.designMatrixTimeIndices;
%create figure for key listening
h = figure(1);
title('Imaris bridge');
set(gcf,'KeyPressFcn',@keyinput);
%cache values used by addsurfacestosurpass
setappdata(h,'numVertices',dataFile.numVertices);
setappdata(h,'numTriangles',dataFile.numTriangles);
setappdata(h,'timeIndices',dataFile.timeIndex);
setappdata(h,'normals',dataFile.normals);
setappdata(h,'vertices',dataFile.vertices);
setappdata(h,'triangles',dataFile.triangles);

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




surfaceClassicationIndex_ = 0;

%Init so it has global scope
classifier = [];
printSelectionInstructions();

    %Get indices of surfaces sorted by distance to point specified
    function [indices] = getsurfacesnearpoint(point)       
        %Get indices of surfaces in current TP 
        currentTPIndices = find(xImarisApp.GetVisibleIndexT == designMatrixTimeIndices); 
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
        func_addsurfacestosurpass(xImarisApp,dataFile,1,xSurfaceToClassify,imarisIndices(index));
    end

    function [] = printSelectionInstructions()
       fprintf('Manual selection mode: \n');
       fprintf('1: position first 2 crosshairs \n');
       fprintf('2: position 3rd crosshair \n');
       fprintf('3: position 3rd crosshair \n');
       fprintf('4: Step backwards through preview \n');
       fprintf('5: Step forwards through preview \n');
       fprintf('Interactive Learning selection mode: \n');
       fprintf('q: label next example at current timepoint \n');
       fprintf('w: classify and visualize all surfaces at current TP \n');
       fprintf('e: classify and visualize all surfaces at all timepoints \n');
       fprintf('Controls for selecting preview cell (both modes): \n');
       fprintf('y: Yes, this is a cell of interest \n');
       fprintf('n: No, this is not a cell of interest \n');
    end

%Controls
    function [] = keyinput(~,~)
        key = get(gcf,'CurrentCharacter');                        
        if strcmp(key,'1') %position crosshairs orthagonal to view            
            %disable active learning mode
            surfaceClassicationIndex_ = 0;
            xClip1.SetVisible(true);
            xClip2.SetVisible(true);
            positioncrosshairs();
            printSelectionInstructions();
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
        elseif strcmp(key,'4') %preview previous surface
            manualPreviewIndex = max(1,manualPreviewIndex - 1);
            updatepreviewsurface();
        elseif strcmp(key,'5') %preview next surface
            manualPreviewIndex = min(length(closestSurfaceIndices),manualPreviewIndex+1);
            updatepreviewsurface();   
        %%%%%%%%% Active Learning %%%%%%%%%
        elseif strcmp(key,'q')
            if (isempty(coiIndices) || isempty(ncoiIndices))
               error('Must manually select 2 small populations of cells to train classifier'); 
            end
            %disable manual selection mode
            manualPreviewIndex = 0;
            % Enter active learning mode: begin presenting unlabelled examples at current time point            
            classifier = retrain(3);
            presentNextExample();
        elseif strcmp(key,'w') % Classify and visualize all instances at current time point
            fprintf('Classifying all surfaces at current time point...\n');
            predictCurrentTP();
        elseif strcmp(key,'e') % Classify all    
            fprintf('Classifying all surfaces...\n');
            predictAll();
        %%%%%%% Add preview surface to cells of interest or not cells of interest %%%%%%%%%
        elseif strcmp(key,'y')
            if surfaceClassicationIndex_ ~= 0 && manualPreviewIndex == 0
                %Yes the currently presented instance show be laballed as a T cell
                coiIndices = unique([coiIndices; surfaceClassicationIndex_]);
                dataFile.coiIndices = coiIndices;
                classifier = retrain(3);
                presentNextExample();
            elseif surfaceClassicationIndex_ == 0 && manualPreviewIndex ~= 0
                coiIndices = unique([coiIndices; closestSurfaceIndices(manualPreviewIndex)]);
                dataFile.coiIndices = coiIndices;
            else
                error('Inconsistent manual selection vs active learning internal state');
            end            
            fprintf('total selected cells of interest: %i\n',length(coiIndices));
            fprintf('total selected not cells of interest: %i\n',length(ncoiIndices));
        elseif strcmp(key,'n')
            %Yes the currently presented instance show be laballed as a T cell
            if surfaceClassicationIndex_ ~= 0 && manualPreviewIndex == 0
                ncoiIndices = unique([ncoiIndices; surfaceClassicationIndex_]);
                dataFile.ncoiIndices = ncoiIndices;
                classifier = retrain(3);
                presentNextExample();
            elseif surfaceClassicationIndex_ == 0 && manualPreviewIndex ~= 0
                ncoiIndices = unique([ncoiIndices; closestSurfaceIndices(manualPreviewIndex)]);
                dataFile.ncoiIndices = ncoiIndices;               
            else
                error('Inconsistent manual selection vs active learning internal state');
            end 
            fprintf('total selected cells of interest: %i\n',length(coiIndices));
            fprintf('total selected not cells of interest: %i\n',length(ncoiIndices));
        end      
    end

    function [cls] = retrain(numLearners)
        %train classifier
        cls = trainClassifier([features(coiIndices,:); features(ncoiIndices,:)],...
            [ones(length(coiIndices),1); zeros(length(ncoiIndices),1)],numLearners);
    end

    function [] = predictAll()
        classifier = retrain(100);
        %delete all predeicted surfaces from imaris
        xPopulationSurface.RemoveAllSurfaces;
        %run classifier on instances at current TP
        pred = classify( classifier, features, ones(size(designMatrixTimeIndices),'logical'),...
            coiIndices, ncoiIndices);
        dataFile.coiPred = find(pred);
        func_addsurfacestosurpass(xImarisApp,dataFile,40, xPopulationSurface,imarisIndices(logical(pred)));
  
    end

    function [] = predictCurrentTP()
        classifier = retrain(20);
        %delete all predeicted surfaces from imaris
        xPopulationSurface.RemoveAllSurfaces;
        %run classifier on instances at current TP
        currentTPMask = xImarisApp.GetVisibleIndexT == designMatrixTimeIndices;
        currentTPPred = classify( classifier, features, currentTPMask, coiIndices, ncoiIndices);
        
        %generate mask for predicted cells of interest at current TP
        currentTPIndices = find(currentTPMask);
        coiAtCurrentTPIndices = currentTPIndices(currentTPPred);
        %send to Imaris
        func_addsurfacestosurpass(xImarisApp,dataFile,100, xPopulationSurface,imarisIndices(coiAtCurrentTPIndices));
    end

    function [] = presentNextExample()
        %find most informative example at this time point (that arent
        %already labelled)
        unlabelledAtCurrentTP = xImarisApp.GetVisibleIndexT == designMatrixTimeIndices;
        unlabelledAtCurrentTP([coiIndices; ncoiIndices]) = 0;
        [~, currentTPPredValue] = classify( classifier, features, unlabelledAtCurrentTP,coiIndices,ncoiIndices);
        
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

