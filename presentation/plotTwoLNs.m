clear
interpN = 1000;

% Other LN-the roudned one 
points_test_ln = load('test_media_ln_points.csv');
points_train_ln = load('training_inguinal_ln_points.csv');

theta = -315; 
R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
rotated = points_test_ln(:, 1:2) * R;
points_test_ln(:, 1:2) = rotated;

%axes start at 0
points_test_ln = points_test_ln - repmat(min(points_test_ln),size(points_test_ln,1),1);
points_train_ln = points_train_ln - repmat(min(points_train_ln),size(points_train_ln,1),1);


%interpolate surface

[xSurf_train, ySurf_train] = meshgrid(linspace(min(points_train_ln(:,1)),max(points_train_ln(:,1)),interpN),...
    linspace(min(points_train_ln(:,2)),max(points_train_ln(:,2)),interpN));
[xSurf_test, ySurf_test] = meshgrid(linspace(min(points_test_ln(:,1)),max(points_test_ln(:,1)),interpN),...
    linspace(min(points_test_ln(:,2)),max(points_test_ln(:,2)),interpN));


zVals_train = dtInterp(delaunay(points_train_ln(:,1),points_train_ln(:,2)), points_train_ln(:,1),points_train_ln(:,2),points_train_ln(:,3),xSurf_train,ySurf_train);
zVals_test = dtInterp(delaunay(points_test_ln(:,1),points_test_ln(:,2)), points_test_ln(:,1),points_test_ln(:,2),points_test_ln(:,3),xSurf_test,ySurf_test);


figure(1)
h = surf(xSurf_train, ySurf_train, -zVals_train);
colormap viridis
set(h, 'edgecolor','none')
axis equal
view(-40,15)
%sweet lighitng
lightangle(-60, 40)
h.FaceLighting = 'gouraud';
h.AmbientStrength = 0.15;
h.DiffuseStrength = 1;
h.SpecularStrength = 0.0;
h.SpecularExponent = 6;
alpha(1)
xlabel('um')

print('train LN','-dtiff')
print -painters -depsc train_ln.eps

% figure(2)
h = surf(xSurf_test, ySurf_test, -zVals_test);
colormap viridis
set(h, 'edgecolor','none')
axis equal
view(-40,15)
%sweet lighitng
lightangle(-60, 40)
h.FaceLighting = 'gouraud';
h.AmbientStrength = 0.15;
h.DiffuseStrength = 1;
h.SpecularStrength = 0.0;
h.SpecularExponent = 6;
alpha(1)
xlabel('um')

print('test LN','-dtiff')
print -painters -depsc test_ln.eps

% ax = gca;
% ax.XTick = [];
% ax.YTick = [];
% ax.ZTick = [];
% axis off
% 
% 
% view(-30,25)


