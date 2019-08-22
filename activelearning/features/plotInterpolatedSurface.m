function [  ] = plotInterpolatedSurface( points, bounds )
%interpolate surface
interpN = 1000;
[xSurf, ySurf] = meshgrid(linspace(min(points(:,1)),max(points(:,1)),interpN),...
    linspace(min(points(:,2)),max(points(:,2)),interpN));
% [xSurf, ySurf] = meshgrid(linspace(bounds(1,1),bounds(1,2),interpN),...
%     linspace(bounds(2,1),bounds(2,2),interpN));

zVals = dtInterp(delaunay(points(:,1),points(:,2)), points(:,1),points(:,2),points(:,3),xSurf,ySurf);


figure(1)
h = surf(xSurf,ySurf,-zVals);

colormap viridis
set(h, 'edgecolor','none')
% view(4,46)

ax = gca;
ax.XTick = [];
ax.YTick = [];
ax.ZTick = [];
axis off

%sweet lighitng
lightangle(60,60)
h.FaceLighting = 'gouraud';
h.AmbientStrength = 0.15;
h.DiffuseStrength = 1;
h.SpecularStrength = 0.0;
h.SpecularExponent = 6;
% print('interpolated LN figure','-dtiff')
alpha(0.75)

end

