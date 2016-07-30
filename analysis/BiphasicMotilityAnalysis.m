clear
load ('DCs_tracks.mat');

figure(1)
clf
%plot number of tracked cells at each time point
allData = cell2mat(tracks);
histogram([allData.t], range([allData.t])+1)

deltaTh = deltaTmin / 60;
%sliding window on each track
%look at only initial balistic regeime
tMinIndex = 20;
tMaxIndex = 55;
deltaTIndices = 1:tMaxIndex - tMinIndex;
deltaTIndices = 1:30;
%each entry is a vector of values for a single deltaT
displacementsSq = cell(length(deltaTIndices),1 );
for trackIndex = 1:length(tracks)
% trackIndex = 8;
    track = tracks{trackIndex};
    if track(1).t + 1 > tMaxIndex
        continue; %track starts after window ends
    elseif  track(end).t + 1 < tMinIndex
        continue; %track ends before window starts
    end
    firstTimeIndex = find([track.t] == tMinIndex - 1);
    lastTimeIndex = find([track.t] == tMaxIndex - 1);
    if isempty (firstTimeIndex)
        firstTimeIndex = track(1).t + 1;
    end
    if isempty (lastTimeIndex)
        lastTimeIndex = track(end).t + 1;
    end
    for dt = deltaTIndices
        trackDisplacementsSq = zeros (lastTimeIndex-dt,1);
        for windowStartIndex = firstTimeIndex:lastTimeIndex-dt
            trackDisplacementsSq(windowStartIndex) = sum((track(windowStartIndex).xyz - track(windowStartIndex+dt).xyz).^2);
        end
        displacementsSq{dt} = [displacementsSq{dt}; trackDisplacementsSq];
    end
    
end
deltaTs = deltaTIndices * deltaTh;
avgDisp = cellfun(@mean,displacementsSq);
stdError = cellfun(@std,displacementsSq) ./ sqrt (cellfun(@length,displacementsSq));
errorbar([0 deltaTs], [0 avgDisp'], [0 stdError'], 'ko')
axis tight
%least squares weights based on 
lsWeights = 1./ stdError.^2;
%fit power law
fun = @(b,deltaT) b(1)*deltaT.^b(2);
b0 = [1000,1];

cutoff1 = 10;
cutoff2 = 22;
nlm1 = fitnlm(deltaTs(1:cutoff1),avgDisp(1:cutoff1)',fun,b0,'Weight',lsWeights(1:cutoff1));
fitXVals1 = linspace(0,deltaTs(cutoff1),1000);
nlm2 = fitnlm(deltaTs(cutoff1+1:cutoff2),avgDisp(cutoff1+1:cutoff2)',fun,b0,'Weight',lsWeights(cutoff1+1:cutoff2));
fitXVals2 = linspace(deltaTs(cutoff1+1),deltaTs(cutoff2),1000);
fitYVals1 = predict(nlm1,fitXVals1');
fitYVals2 = predict(nlm2,fitXVals2');
hold on 
plot(fitXVals1,fitYVals1,'b-')
plot(fitXVals2,fitYVals2,'g-')
% plot([0 deltaTs(end)],[0 avgDisp(end)],'r--')
hold off
legend('Data','Supradiffusive regeime','subdiffusive regeime')
xlabel('\Deltat (h)')
ylabel('Displacement.^2 (\mum^2)')

ci1 = coefCI(nlm1);
ci2 = coefCI(nlm2);
annotation('textbox',[.2 .6 .3 .3],'String',...
    sprintf('Fit 1: x^2 = %0.0f*\\Deltat^{%0.2f}\n95 CI: %0.2f-%0.2f',...
    nlm1.Coefficients.Estimate(1),nlm1.Coefficients.Estimate(2),ci1(2,1),ci1(2,2)),'FitBoxToText','on','FontSize',20);
annotation('textbox',[.2 .5 .3 .3],'String',...
    sprintf('Fit 2: x^2 = %0.0f*\\Deltat^{%0.2f}\n95 CI: %0.2f-%0.2f',...
    nlm2.Coefficients.Estimate(1),nlm2.Coefficients.Estimate(2),ci2(2,1),ci2(2,2)),'FitBoxToText','on','FontSize',20);


exportPlot('Subsequent Motility Full')
