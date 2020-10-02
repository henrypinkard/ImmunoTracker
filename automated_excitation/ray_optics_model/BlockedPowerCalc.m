%DEPRECATED
function [powerFraction] = BlockedPowerCalc (theta)
% theta is surface normal angle

%calculate the amount of power blocked out of excitaiton
%cone based on normal angle

%back aperture coordinates go from -1 to 1
%FWHM = 2.36*sigma
FWHM = 1.2;
sigma = FWHM / 2.36;
alpha = 60; %NA angle in degrees

fullpower = integral2(@(x,y) 1 / (2.*pi.*sigma.*sigma) .* exp(-(x.^2 + y.^2) / (2.*sigma.^2)),...
    -1,1,-1,1);

%solve for partial power based on normal angle theta
%find x lim between 0 and 1
xlim = sqrt( 1 / sind(alpha).^2 - 1 ) * tand(90 - theta);

power = integral2(@(x,y) 1 / (2.*pi.*sigma.*sigma) .* exp(-(x.^2 + y.^2) / (2.*sigma.^2)),...
    -1,1,-1,xlim);
powerFraction = power / fullpower;
end