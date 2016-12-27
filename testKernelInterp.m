clear
points = load('6-6 updated.txt');

%axes start at 0
points = points - repmat(min(points),size(points,1),1);



%interpolate surface
interpN = 200;
[xSurf, ySurf] = meshgrid(linspace(min(points(:,1)),max(points(:,1)),interpN),...
    linspace(min(points(:,2)),max(points(:,2)),interpN));
zVals = dtInterp(delaunay(points(:,1),points(:,2)), points(:,1),points(:,2),points(:,3),xSurf,ySurf);

%plot interpolated surface
clear
figure(3)
load('1000xLNInterp.mat')
h = surf(xSurf,ySurf,-zVals + 400);
colormap jet
set(h, 'edgecolor','none')
view(4,46)

gridx1 = linspace(min(points(:,1)),max(points(:,1)));
gridx2 = linspace(min(points(:,2)),max(points(:,2)));
[x1,x2] = meshgrid(gridx1, gridx2);
x1 = x1(:);
x2 = x2(:);
xi = [x1 x2];

ksdensity(points(:,1:2),xi);


