%use empirically calculated depth lists and infer the mean free path
clear
radiusOfCurvature = 600;

%Assume that depth lists were calculated from the top of a curved pLN
verticalDistance = 0:25:250;

MFPs = 100:25:250;

relPower = zeros(1,length(verticalDistance));

for mfp = 1: length(MFPs)
    for d = 1:length(verticalDistance)
        fprintf('distance: %i\n',verticalDistance(d));
        relPower(mfp, d) = PowerAttentuationCalc(0,verticalDistance(d),MFPs(mfp),radiusOfCurvature);   
    end
end

%multiply by exponentially increasing power to try to find constant
%excitation with depth

%C -- 40 e ^0.008 x
%M -- 7 e ^ 0.014 x
%C cleared -- 0.006 -- 125 mfp
%M cleared -- 0.0045 -- 200 mfp
inputPower = exp(0.0045 * verticalDistance);
% inputPower = exp(0.014 * verticalDistance);

effExcitation = relPower .* repmat(inputPower,size(relPower,1),1);
plot(verticalDistance,effExcitation)
legend(cellstr(num2str(MFPs', '%d') ))



%810 (MT) MFP ~ 30
%870 (Cham) MFP ~ 90
