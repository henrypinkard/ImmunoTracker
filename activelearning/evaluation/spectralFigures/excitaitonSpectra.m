clear 
load('2p1pexcitationspectra.mat')

%clip spectra
cfp1p = cfp1p(lamcfp1p > 350 & lamcfp1p < 560);
gfp1p = gfp1p(lamgfp1p > 350 & lamgfp1p < 560);
lamgfp1p = lamgfp1p(lamgfp1p > 350 & lamgfp1p < 560);
lamcfp1p = lamcfp1p(lamcfp1p > 350 & lamcfp1p < 560);

cfp2p = cfp2p(lamcfp2p > 700);
gfp2p = gfp2p(lamgfp2p > 700);
lamgfp2p = lamgfp2p(lamgfp2p > 700);
lamcfp2p = lamcfp2p(lamcfp2p > 700);




figure(1)
subplot(2,1,1)
h1 = area (lamcfp2p, cfp2p);
hold on
h2 = area (lamgfp2p, gfp2p);
hold off
legend('CFP 2-photon','GFP 2-photon')
ylabel('Relative excitation')
xlabel('Wavelength (nm)')
set(h1,'EdgeColor', 'none','FaceColor', [0 0 1]);
set(h2,'EdgeColor', 'none', 'FaceColor', [0 1 0]);
alpha(0.7)

subplot(2,1,2)
h1 = area (lamcfp1p, cfp1p);
hold on
h2 = area (lamgfp1p, gfp1p);
hold off
legend('CFP 1-photon','GFP 1-photon')
ylabel('Relative excitation')
xlabel('Wavelength (nm)')
set(h1,'EdgeColor', 'none','FaceColor', [0 0 1]);
set(h2,'EdgeColor', 'none', 'FaceColor', [0 1 0]);
alpha(0.7)

