function [ classifier ] = trainClassifier( trainData, trainLabels, numLearners )
% train a classifier using given training data and labels. Return some kind
% of classifier object that will later be passed to classify.m for
% classificaiton

%train ensemble
classifier = cell(numLearners,1);
for j = 1: numLearners
    fprintf('Training classifier %i of %i\n',j,numLearners);
    classifier{j} = trainNN();
end


function [singleNN] = trainNN()
%Determine number of hiddens by cross validation
% paramVals = [1 4 8 12 16 20 25 30 40 50 60 80 100];
% numNetworks = 10;
% perf = zeros(length(paramVals),numNetworks);
% for networkIndex = 1:numNetworks
%     for index = 1:length(paramVals)
%         param = paramVals(index);
%         net = patternnet(param,'trainscg','crossentropy' );
%         net.trainParam.lr_dec = 0.05;
%         x = trainData';
%         t = trainLabels';
%         [net,tr] = train(net,x,t );
%         
%         %evaluate on test data
%         testX = x(:,tr.testInd);
%         testT = t(:,tr.testInd);
%         testY = net(testX);
%         % testClasses = testY > 0.5
%         [c,cm] = confusion(testT,testY);
%         perf(index,networkIndex) = c;
%     end
% end
% figure(2)
% perfAvg = mean(perf,2);
% perfSTD = std(perf,0,2);
% errorbar(paramVals,perfAvg,perfSTD);
% 
% plotconfusion(testT,testY)
% plotroc(testT,testY)



%%train neural net on all data and predict unseens
    %best number of hiddens = 20;
    net = patternnet(12,'trainscg','crossentropy' );
    net.divideParam.testRatio = 0;
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
%     net.trainParam.lr_dec = 0.05;
    % net.trainParam.lr = 0.1;
    % net.trainParam.mc = 1;
    net.trainParam.showWindow = false;
    net.trainParam.showCommandLine = false;
    
%     numZeros = sum(trainLabels == 0);
%     numOnes = sum(trainLabels);
%     if numOnes < numZeros
%         %more 0s than 1s
%         zeroIndices = find(trainLabels==0);
%         indPerm = randperm(numZeros);
%         toRemoveInd = zeroIndices(indPerm(1:numZeros-numOnes));
%         trainLabels(toRemoveInd) = [];
%         trainData(toRemoveInd,:) = [];
%     else
%         %more 1s than 0s
%         oneIndices = find(trainLabels==0);
%         indPerm = randperm(numOnes);
%         toRemoveInd = oneIndices(indPerm(1:numOnes-numZeros));
%         trainLabels(toRemoveInd) = [];
%         trainData(toRemoveInd,:) = []; 
%     end


    [singleNN,tr] = train(net,trainData',trainLabels','useParallel','yes');
end

end
