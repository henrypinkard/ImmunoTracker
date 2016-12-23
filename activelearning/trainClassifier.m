function [ neuralNet ] = trainClassifier( trainData, trainLabels )

%Determine number of hiddens by cross validation
% nHiddens = [1 5 10 15 20 50 100 120]
% perf = zeros(size(nHiddens));
% for index = 1:length(nHiddens)
%     nHidden = nHiddens(index)
%     net = patternnet(nHidden,'trainscg','crossentropy' );
%     x = features(dataToUseMask,:)';
%     t = labels';
%     [net,tr] = train(net,x,t );
%     
%     %evaluate on test data
%     testX = x(:,tr.testInd);
%     testT = t(:,tr.testInd);
%     testY = net(testX);
%     % testClasses = testY > 0.5
%     [c,cm] = confusion(testT,testY);
%     perf(index) = c;
% end
% plot(nHiddens,perf,'o-')
% 
% plotconfusion(testT,testY)
% plotroc(testT,testY)


%%train neural net on all data and predict unseens
%best number of hiddens = 20;
net = patternnet(20,'trainscg','crossentropy' );
net.divideParam.testRatio = 0;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.2;
net.trainParam.showWindow = false;
net.trainParam.showCommandLine = false; 
[neuralNet,tr] = train(net,trainData',trainLabels','useParallel','yes');

end

