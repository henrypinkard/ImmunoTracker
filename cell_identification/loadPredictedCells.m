function [  ] = loadPredictedCells(  )

[xImarisApp] = xtConnect();

% 44-45: [  -2, 1102,  587]
% 48-49: [-10 692 857]

path =  '/Users/henrypinkard/Desktop/featurized_sruface_candidates/';

[imarisIndices, coiIndices, ncoiIndices, timeIndices,...
            otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '45', 'xcr1');
predictAll(false, 'XCR1');

% [imarisIndices, coiIndices, ncoiIndices, timeIndices,...
%             otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '44', 'rfp');
% predictAll(false, 'RFP');

[imarisIndices, coiIndices, ncoiIndices, timeIndices,...
            otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '45', 'vpd');
predictAll(false, 'VPD');

[imarisIndices, coiIndices, ncoiIndices, timeIndices,...
            otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '45', 'gfp');
predictAll(false, 'GFP');


offsetZ = 40;
% 
% [imarisIndices, coiIndices, ncoiIndices, timeIndices,...
%             otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '48', 'xcr1');
% spotCenters(:, 3) = spotCenters(:, 3) + offsetZ;
% predictAll(false, 'XCR1');
% 
% [imarisIndices, coiIndices, ncoiIndices, timeIndices,...
%             otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '48', 'gfp');
% spotCenters(:, 3) = spotCenters(:, 3) + offsetZ;
% predictAll(false, 'GFP');
%         
% [imarisIndices, coiIndices, ncoiIndices, timeIndices,...
%             otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '48', 'rfp');
% spotCenters(:, 3) = spotCenters(:, 3) + offsetZ;
% predictAll(false, 'RFP');
% 
% [imarisIndices, coiIndices, ncoiIndices, timeIndices,...
%             otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, '48', 'vpd');
% spotCenters(:, 3) = spotCenters(:, 3) + offsetZ;
% predictAll(false, 'VPD');

    function [imarisIndices, coiIndices, ncoiIndices, designMatrixTimeIndices,...
            otherCOIFeatures, otherNCOIFeatures, features, spotCenters] = initialize(path, fileNum, populationName)
        fullpath = strcat(path, fileNum, '_', populationName, '_candidates.mat');
        fprintf('opening data from: %s\n', fullpath);
        dataFile = matfile(fullpath,'Writable',true);
        %cache values for performance
        features = dataFile.features;
        
        otherCOIFeatures = zeros(0, size(features,2));
        otherNCOIFeatures = zeros(0, size(features,2));
        files = dir(strcat(path, '*', populationName,'*.mat'));
        for f = files'
            load(strcat(path, f.name),'coiIndices');
            load(strcat(path, f.name),'ncoiIndices');
            if exist('coiIndices','var') || exist('ncoiIndices','var')
                fprintf('Found existing labels from %s\n', f.name);
                otherDataFile = matfile(strcat(path,f.name),'Writable',false);
                otherFeatures = otherDataFile.features;
                otherCOIFeatures = [otherCOIFeatures; otherFeatures(otherDataFile.coiIndices, :) ];
                otherNCOIFeatures = [otherNCOIFeatures; otherFeatures(otherDataFile.ncoiIndices, :) ];
                clear coiIndices
                clear ncoiIndices
            else
                fprintf('No existing labels from %s\n', f.name);
            end
        end
        
        %createLabels for cells of interest
        if (~any(strcmp('coiIndices',who(dataFile))))
            dataFile.coiIndices = [];
        end
        if (~any(strcmp('ncoiIndices',who(dataFile))))
            dataFile.ncoiIndices = [];
        end
        
        imarisIndices = dataFile.imarisIndices;
        coiIndices = dataFile.coiIndices; %for including ground truth labels
        ncoiIndices = dataFile.ncoiIndices;
        designMatrixTimeIndices = dataFile.designMatrixTimeIndices;
        %create figure for key listening
        
        % spotCenters = getappdata(h,'spotCenters');
        featureIndices = cell2mat(cellfun(@(n) find(strcmp(dataFile.rawFeatureNames, n)),...
            {'Position X', 'Position Y', 'Position Z'}, 'UniformOutput', false));
        spotCenters = dataFile.rawFeatures(:, featureIndices) ;
        %TODO normalized cut roi centers
        
    end

    function [ xImarisApp ] = xtConnect()
        xtIndex = 0;
        javaaddpath('./ImarisLib.jar')
        vImarisLib = ImarisLib;
        xImarisApp = vImarisLib.GetApplication(xtIndex);
        if (isempty(xImarisApp))
            error('Wrong imaris index');
        end
        
        xSurpass = xImarisApp.GetSurpassScene;
    end

    function [] = predictAll(asSurfaces, name)
        classifier = retrain(100);
        %delete all predeicted surfaces from imaris
        %run classifier on instances at current TP
        pred = classify( classifier, features, ones(size(timeIndices),'logical'),...
            coiIndices, ncoiIndices, 0);
        dataFile.coiPred = find(pred);
        if asSurfaces
            func_addsurfacestosurpass(xImarisApp,dataFile,40, xPopulationSurface,imarisIndices(logical(pred)));
        else
            addSpotsToSurpass(xImarisApp, imarisIndices(logical(pred)), name, spotCenters, timeIndices)
        end
        
    end

    function [cls] = retrain(numLearners)
        %train classifier
        cls = trainClassifier([ otherCOIFeatures; otherNCOIFeatures],...
            [ones(size(otherCOIFeatures, 1), 1);...
            zeros(size(otherNCOIFeatures, 1), 1)],numLearners, 0);
    end

end