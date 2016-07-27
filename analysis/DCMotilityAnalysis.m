clear
%connect to imaris to get DC tracks
javaaddpath xt/ImarisLib.jar
vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(0);
if (isempty(xImarisApp))
    msgbox('Wrong imaris index');
    return;
end
xSurpass = xImarisApp.GetSurpassScene;
surfaces = xImarisApp.GetFactory.ToSurfaces(xImarisApp.GetSurpassSelection);

%convert Imaris tracks to something more sensible
edges = surfaces.GetTrackEdges; %0 based
%make adjacency matrix
adjMat = zeros(max(edges(:))+1);
adjMat(sub2ind(size(adjMat),edges(:,1)+1,edges(:,2)+1)) = 1;
adjMat = adjMat + adjMat';
gr = graph(adjMat);
trackSurfaceIndices = conncomp(gr,'OutputForm','cell');

tracks = cell(size(trackSurfaceIndices));
for trackIndex = 1:length(tracks)
   surfaceIndices = trackSurfaceIndices{trackIndex} - 1; 
   xyzt = zeros(length(surfaceIndices),1);
   for i = 1:length(surfaceIndices)
      xyzt(i,4) = surfaces.GetTimeIndex(surfaceIndices(i));
      xyzt(i,1:3) = surfaces.GetCenterOfMass(surfaceIndices(i));
   end
   %sort by time
   [~,I] = sort(xyzt(:,4));
   xyzt = xyzt(I,:);
   tracks{trackIndex} = xyzt;
end

figure(1)
%plot number of tracked cells at each time point
allData = cell2mat(tracks');
histogram(allData(:,4), range(allData(:,4))+1)

deltaTmin = xImarisApp.GetDataSet.GetTimePointsDelta / 60;
deltaTh = deltaTMin / 60;
%get displacements for each track 
getDispSq = @(xyzt) [sum((xyzt(:,1:3) - repmat(xyzt(1,1:3),size(xyzt,1),1)).^2,2) xyzt(:,4)];
tracksDispSq = cellfun(getDispSq,tracks,'UniformOutput',false);
figure(1)
singleTrackDispSq = tracksDispSq{1};
plot(deltaTh * singleTrackDispSq(:,2),singleTrackDispSq(:,1))
hold on
for i = 2:length(trackDispSq)
    singleTrackDispSq = tracksDispSq{i};
    plot(deltaTh * singleTrackDispSq(:,2),singleTrackDispSq(:,1),'.-')
end
hold off
ylabel('Displacement^2 (\mum^2)')
xlabel('Time (h)')
exportPlot('DC displacementSq')


%velocity vs time
getVelocity = @(xyzt) [sqrt(sum((xyzt(2:end,1:3) - xyzt(1:end-1,1:3)).^2,2))/deltaTh xyzt(2:end,4)];
velocities = cellfun(getVelocity,tracks,'UniformOutput',false);
figure(2)
avgSpeed = zeros(max(allData(:,4))+1,1);
stdError = zeros(max(allData(:,4))+1,1);
for t = 1:length(avgSpeed)
    %find velocity for all tracks at timepoint
    vAtTP = cellfun(@(track) track(find(track(:,2)==t,1),1),velocities,'UniformOutput',false);
    vAtTP = cell2mat(vAtTP(~cellfun(@isempty,vAtTP)));  %ignore missing tracks
    avgSpeed(t) = mean(vAtTP);
    stdError(t) = std(vAtTP) / sqrt(length(vAtTP)-1);
end
errorbar((1:length(avgSpeed))*deltaTh,avgSpeed,stdError)
ylabel('Velocity (\mum/h)')
xlabel('Time (h)')
exportPlot('DC velocities')