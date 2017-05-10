%change to current dir
cd(fileparts(mfilename('fullpath')))

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

% plot points
numPoints = 80;
noiseFactor = 8;

%figure 1-- 3d points
figure (1)
% GFP
ratio = intensityTriple(gfp);
ratio = ratio ./ norm(ratio);
xyz = repmat(ratio,numPoints,1);
intensity = randi(255,numPoints,1);
xyz = xyz .* repmat(intensity,1,3);
%add noise
xyzG =xyz + normrnd(0,xyz / noiseFactor);
scatter3(xyzG(:,1),xyzG(:,2),xyzG(:,3), 'filled',...
    'MarkerEdgeColor',[0 0.7 0], 'MarkerFaceColor', [0 0.7 0] )
hold on

% YFP
ratio = intensityTriple(yfp);
ratio = ratio ./ norm(ratio);xyz = repmat(ratio,numPoints,1);
intensity = randi(255,numPoints,1);
xyz = xyz .* repmat(intensity,1,3);
%add noise
xyzY =xyz + normrnd(0,xyz / noiseFactor);
scatter3(xyzY(:,1),xyzY(:,2),xyzY(:,3), 'filled',...
    'MarkerEdgeColor',[1 0.7 0.3], 'MarkerFaceColor',[1 0.7 0.3] )

% CFP
ratio = intensityTriple(cfp);
ratio = ratio ./ norm(ratio);
xyz = repmat(ratio,numPoints,1);
intensity = randi(255,numPoints,1);
xyz = xyz .* repmat(intensity,1,3);
%add noise
xyzB =xyz + normrnd(0,xyz / noiseFactor);
scatter3(xyzB(:,1),xyzB(:,2),xyzB(:,3), 'filled',...
    'MarkerEdgeColor',[0 0 1], 'MarkerFaceColor', [0 0 1] )

hold off

view(27,36)
xlabel('Yellow','FontSize',20)
ylabel('Green','FontSize',20)
zlabel('Blue','FontSize',20)
set(gca,'GridLineStyle','-')
set(gca,'XTickLabel', [])
set(gca,'YTickLabel', [])
set(gca,'ZTickLabel', [])
legend('GFP cells','YFP cells','CFP Cells')
print('3DPoints figure','-depsc')

%figure 2 2d points
figure (2)
plot(xyzG(:,2),xyzG(:,1),'o','MarkerFaceColor',[0 0.7 0],'MarkerEdgeColor',[0 0.7 0])
hold on
plot(xyzB(:,2),xyzB(:,1),'o','MarkerFaceColor',[0 0 1],'MarkerEdgeColor',[0 0 1])
plot(xyzY(:,2),xyzY(:,1),'o','MarkerFaceColor',[1 0.7 0.3],'MarkerEdgeColor',[1 0.7 0.3])
set(gca,'XTickLabel', [])
set(gca,'YTickLabel', [])

xlabel('Green','FontSize',20)
ylabel('Yellow','FontSize',20)
hold off
print('2d Points Figure','-depsc')
legend('GFP cells','CFP Cells','YFP Cells')



