clear


% points are from hi number right LN 5-30
points = load('LNPoints.txt');

%axes start at 0
points = points - repmat(min(points),size(points,1),1);


%plot interpolation points in 3D space
figure(1); 
markerSize = 50;
scatter3(points(:,1),points(:,2),-points(:,3) + 400,markerSize, -points(:,3),'filled')
% set(gca,'zdir','reverse')
view(4,46)
% colomap and bar
colormap(viridis);
colorbar;
xlabel('X')
ylabel('Y')
zlabel('Z')
print('interp points figure','-depsc', '-painters')


%plot 2DPoints projected and delaunay triangualtion
tris = delaunay(points(:,1),points(:,2));
pairIndices = cellfun(@(row) mat2cell(combnk(row,2),[1 1 1],2), num2cell(tris,2),'UniformOutput',0);
%flatten
pairIndices = cell2mat(cellfun(@(entry) cell2mat(entry),pairIndices,'UniformOutput',0));

triX = [points(pairIndices(:,1),1) points(pairIndices(:,2),1)];
triY = [points(pairIndices(:,1),2) points(pairIndices(:,2),2)];

figure(2)
%plot delauney triangulation and vertices
plot(triX',triY','k-' )
hold on
scatter(points(:,1),points(:,2),markerSize, -points(:,3),'filled')
colormap(jet);
hold off

xlabel('X')
ylabel('Y')
print('DT figure','-depsc', '-painters')


%interpolate surface
% interpN = 1000;
% [xSurf, ySurf] = meshgrid(linspace(min(points(:,1)),max(points(:,1)),interpN),...
%     linspace(min(points(:,2)),max(points(:,2)),interpN));
% zVals = dtInterp(delaunay(points(:,1),points(:,2)), points(:,1),points(:,2),points(:,3),xSurf,ySurf);

%plot interpolated surface
clear
figure(3)
load('1000xLNInterp.mat')
h = surf(xSurf,ySurf,-zVals + 400);
colormap jet
set(h, 'edgecolor','none')
view(4,46)

ax = gca;
ax.XTick = [];
ax.YTick = [];
ax.ZTick = [];
axis off

%sweet lighitng
lightangle(60,60)
h.FaceLighting = 'gouraud';
h.AmbientStrength = 0.3;
h.DiffuseStrength = 0.9;
h.SpecularStrength = 0.4;
h.SpecularExponent = 5;
print('interpolated LN figure','-dtiff')





