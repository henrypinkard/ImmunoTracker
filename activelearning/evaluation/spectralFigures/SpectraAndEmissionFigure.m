clear 
load emissionspectra.mat


violet = [387 447];
blue = [463 487];
green = [489 528];
yellow = [531 556];
red = [572 642];
farred = [641 709];

blue = [463 500];
green = [502 528];



%plot rectangles
h1 = area (lambda, cfp);
hold on
h2 = area (lambda, gfp);
h3 = area (lambda, yfp);
hold off
legend('CFP','GFP','YFP')
ylabel('Relative intensity')
xlabel('Wavelength (nm)')
set(h1,'EdgeColor', 'none','FaceColor', [0 0 1]);
set(h2,'EdgeColor', 'none', 'FaceColor', [0 1 0]);
set(h3,'EdgeColor', 'none', 'FaceColor', [1 0.7 0.3]);
alpha(0.7)

%plot filters
r1 = patch([blue(1) blue(1) blue(2) blue(2)], [0 1 1 0] ,'b');
r2 = patch([green(1) green(1) green(2) green(2)], [0 1 1 0] ,'g');
r3 = patch([yellow(1) yellow(1) yellow(2) yellow(2)], [0 1 1 0] ,'y');
r4 = patch([red(1) red(1) red(2) red(2)], [0 1 1 0] ,'r');

filterAlpha = 0.4;
set(r1,'EdgeColor', 'none', 'FaceAlpha', filterAlpha);
set(r2,'EdgeColor', 'none', 'FaceAlpha', filterAlpha);
set(r3,'EdgeColor', 'none','FaceAlpha', filterAlpha);
set(r4,'EdgeColor', 'none', 'FaceAlpha', filterAlpha);

print('emission and filters figure','-depsc', '-painters')
