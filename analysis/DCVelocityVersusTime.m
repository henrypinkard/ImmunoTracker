clear
load ('DCs_tracks.mat');
deltaTh = deltaTmin / 60;
allData = cell2mat(tracks);

%velocity vs time
figure(1)
getVelocity = @(track) sqrt(sum((cat(1,track(2:end).xyz) - cat(1,track(1:end-1).xyz)).^2,2))/deltaTh;
velocities = cellfun(getVelocity,tracks,'UniformOutput',false);
avgSpeed = zeros(max([allData.t]),1);
stdError = zeros(max([allData.t]),1);
for t = 1:length(avgSpeed)
    %find velocity for all tracks at timepoint
    vAtTP = cellfun(@(track,speed) speed(find([track(2:end).t]==t,1)),tracks,velocities,'UniformOutput',false);
    vAtTP = cell2mat(vAtTP(~cellfun(@isempty,vAtTP)));  %ignore missing tracks
    avgSpeed(t) = mean(vAtTP);
    stdError(t) = std(vAtTP) / sqrt(length(vAtTP)-1);
end
errorbar((1:length(avgSpeed))*deltaTh,avgSpeed,stdError)
ylabel('Velocity (\mum/h)')
xlabel('Time (h)')
axis tight
exportPlot('DC velocities')