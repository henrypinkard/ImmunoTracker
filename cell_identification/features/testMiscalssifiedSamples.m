function [] = testMiscalssifiedSamples()
%cnnect to imaris
imarisIndex = 0;
javaaddpath ImarisLib.jar
vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(imarisIndex);
if (isempty(xImarisApp))
    msgbox('Wrong imaris index');
    return;
end
xPreviewSurface = xImarisApp.GetFactory.CreateSurfaces;
xPreviewSurface.SetName('Preview surface');
xSurpassCam = xImarisApp.GetSurpassCamera;
xSurpass = xImarisApp.GetSurpassScene;
xSurpass.AddChild(xPreviewSurface,-1);


filename = '/Users/henrypinkard/Desktop/LNData/CMTMRCandidates.mat';
surfFile = matfile(filename,'Writable',true);
stats = surfFile.stats;
timeIndices = stats(find(strcmp({stats.Name},'Time Index'))).Values;


exceldata = xlsread('/Users/henrypinkard/Google Drive/Code/MATLAB/CS289 project/ElasticNet/6_6_16/Ave_Fin_Probability_B2000.xlsx');
Sample = exceldata(:,1);
Misclassified = exceldata(:,7);
load('/Users/henrypinkard/Google Drive/Code/MATLAB/CS289 project/data/CMTMRFeaturesAndLabels.mat',...
    'imarisIndices','labelledTCell','labelledNotTCell');
iis = imarisIndices([labelledTCell; labelledNotTCell] + 1);
misclassifiedIIs = iis(Sample(Misclassified == 1));
%%which ones are T cells
tCellIIs = imarisIndices(labelledTCell+1);
ntCellIIs = imarisIndices(labelledNotTCell+1);

misclassTCellIIs = misclassifiedIIs(ismember(misclassifiedIIs,tCellIIs));
misclassNTCellIIs = misclassifiedIIs(ismember(misclassifiedIIs,ntCellIIs));
dataIIs = misclassNTCellIIs;
wrongLabelIIs = [];
previewIndex = 1;

%setup bridge


figure(1);
title('Imaris bridge');
set(gcf,'KeyPressFcn',@keyinput);

updatepreviewsurface();
centertopreview();


    function [] = updatepreviewsurface()
        xPreviewSurface.RemoveAllSurfaces; %clear old ones
        func_addsurfacestosurpass(xImarisApp,surfFile,1,xPreviewSurface,dataIIs(previewIndex));
    end

%surpass camera parameters:
%height--distance from the object it is centered on, changes with zoom but not with rotation
%position--camera position, not related to center of rotation
%focus--seemingly not related to anything... (maybe doesn't matter for orthgraphic)
%Fit will fit viewing frame around entire image...which will in turn set the center of rotation to the center of image
    function [] = centertopreview()  
    quatRot = @(theta,axis) [axis / norm(axis) * sin(theta / 2); cos(theta/2)];
        xImarisApp.SetVisibleIndexT(double(timeIndices(dataIIs(previewIndex)+1)));
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


    function [] = keyinput(~,~)
        key = get(gcf,'CurrentCharacter');
        if previewIndex == length(dataIIs)
            %done
            wrongLabelIIs;
            return;
        elseif strcmp(key,'s') 
            fprintf('%i of %i',previewIndex,length(dataIIs));
        elseif strcmp(key,'w') %correctly labeled, next
            previewIndex = previewIndex + 1;
        elseif strcmp(key,'e') %incorrect label, add to set
            %select in file
            wrongLabelIIs = unique([wrongLabelIIs; dataIIs(previewIndex)])
            previewIndex = previewIndex + 1;
        end
            updatepreviewsurface();
            centertopreview();
        
    end
end
