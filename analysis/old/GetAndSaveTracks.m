function [  ] = GetAndSaveTracks( )
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

tracks = cell(size(trackSurfaceIndices'));
for trackIndex = 1:length(tracks)
    surfaceIndices = trackSurfaceIndices{trackIndex} - 1;
    singleTrack = struct('xyz',[],'t',[]);
    for i = 1:length(surfaceIndices)
        singleTrack(i).t = surfaces.GetTimeIndex(surfaceIndices(i));
        singleTrack(i).xyz = surfaces.GetCenterOfMass(surfaceIndices(i));
    end
    %sort by time
    [~,I] = sort([singleTrack.t]);
    singleTrack = singleTrack(I);
    %fill in missing entries
    endIndex = singleTrack(end).t - singleTrack(1).t + 1;
    for tIndex = 2:endIndex
       if singleTrack(tIndex).t ~= singleTrack(tIndex - 1).t + 1
          singleTrack(tIndex+1:end+1) = singleTrack(tIndex:end); %shift back by one
          endIndex = endIndex + 1;
          %fill in missing values
          singleTrack(tIndex).t = singleTrack(tIndex -1).t + 1;
          singleTrack(tIndex).xyz = singleTrack(tIndex+1).xyz *... 
              abs(singleTrack(tIndex+1).t - singleTrack(tIndex).t) / abs(singleTrack(tIndex+1).t - singleTrack(tIndex-1).t)...
              + singleTrack(tIndex-1).xyz...
              * abs(singleTrack(tIndex-1).t - singleTrack(tIndex).t) / abs(singleTrack(tIndex+1).t - singleTrack(tIndex-1).t);
       end
    end
    tracks{trackIndex,1} = singleTrack';
end

deltaTmin = xImarisApp.GetDataSet.GetTimePointsDelta / 60;

%save in same dir as function
fullpath = mfilename('fullpath');
mfn = mfilename;
fullpath = fullpath(1:end-length(mfn));
save(strcat(fullpath,char(surfaces.GetName),'_tracks'),'tracks','deltaTmin');
end

