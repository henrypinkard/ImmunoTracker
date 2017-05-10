clear
figure('Color',[1 1 1])
colored_sphere_figure
numPoints = 50;
noiseFactor = 8;


load emissionspectra.mat

violet = [387 447];
% blue = [463 487];
% green = [489 528];
yellow = [531 556];
red = [572 642];
farred = [641 709];

blue = [463 500];
green = [502 528];

intensityFraction = @(filter, spectrum) sum(spectrum(lambda > filter(1) & lambda < filter(2) )) / sum(spectrum); 
intensityTriple = @(spectrum) [intensityFraction(yellow,spectrum) intensityFraction(green,spectrum) intensityFraction(blue,spectrum) ]; 


ratio = intensityTriple(gfp);
ratio = ratio ./ norm(ratio);
xyz = repmat(ratio,numPoints,1);
intensity = randi(255,numPoints,1);
xyz = xyz .* repmat(intensity,1,3);
%add noise
xyzG =xyz + normrnd(0,xyz / noiseFactor);

% YFP
ratio = intensityTriple(yfp);
ratio = ratio ./ norm(ratio);xyz = repmat(ratio,numPoints,1);
intensity = randi(255,numPoints,1);
xyz = xyz .* repmat(intensity,1,3);
%add noise
xyzY =xyz + normrnd(0,xyz / noiseFactor);

% CFP
ratio = intensityTriple(cfp);
ratio = ratio ./ norm(ratio);
xyz = repmat(ratio,numPoints,1);
intensity = randi(255,numPoints,1);
xyz = xyz .* repmat(intensity,1,3);
%add noise
xyzB =xyz + normrnd(0,xyz / noiseFactor);

%normailzed points
hold on 
xyzGN = bsxfun(@rdivide,xyzG,sqrt(sum(xyzG.^2,2)));
xyzYN = bsxfun(@rdivide,xyzY,sqrt(sum(xyzY.^2,2)));
xyzBN = bsxfun(@rdivide,xyzB,sqrt(sum(xyzB.^2,2)));
scatter3(xyzBN(:,1),xyzBN(:,2),xyzBN(:,3), 'filled',...
    'MarkerEdgeColor',[0 0 1], 'MarkerFaceColor', [0 0 1] )
scatter3(xyzYN(:,1),xyzYN(:,2),xyzYN(:,3), 'filled',...
    'MarkerEdgeColor',[1 0.7 0], 'MarkerFaceColor', [1 0.7 0] )
scatter3(xyzGN(:,1),xyzGN(:,2),xyzGN(:,3), 'filled',...
    'MarkerEdgeColor',[0 0.7 0], 'MarkerFaceColor', [0 0.7 0] )
hold off

view(115,28)
xlabel('')
zlabel('')
ylabel('')
ax = gca;
set(gca,'YTick', []);
set(gca,'XTick', []);
set(gca,'ZTick', []);
set(gca,'LineWidth', 2);



