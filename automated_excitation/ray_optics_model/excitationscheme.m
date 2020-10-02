clear
clc

%Input: surface normal and vertical distance
%output: LN distance--the vertical distance corresponding to the same
%attentuation of light as this one

%make LUTs that give LN distance as a function of normal angle and vertical
%distance 


%LN mfp = 125 (870), 71 (810)
%radius of popliteal LN:
%400 (cleared)
%600 (live)
radiusOfCurvature = 400;
normal = 0:10:90; 
verticalDistance = 0:20:800;
%estimated MFP at 870 (Chameleon) = 90, 810 (Maitai) = 30
MFPs = [120];


extraPowerNeeded = cell(length(MFPs),1);
for m = 1:length(MFPs)
    transmitLUT = zeros(length(normal),length(verticalDistance));
    fprintf('MFP %i\n',MFPs(m));
    for n = 1:length(normal)
        for d = 1:length(verticalDistance)
            fprintf('normal angle: %i    distance: %i\n',normal(n),verticalDistance(d));
            %calculate transmittance
            tic
            transmitLUT(n,d) = PowerAttentuationCalc(normal(n),verticalDistance(d),MFPs(m),radiusOfCurvature);
            toc
        end
    end
    transmitLUT = 1./ transmitLUT;
    extraPowerNeeded{m} = transmitLUT;
    fprintf('\n\n');
end


fprintf('{');
for i = 1:size(transmitLUT,1)
    fprintf('{');
   for j = 1:size(transmitLUT,2)
      fprintf('%f,',transmitLUT(i,j));
   end
   fprintf('%f',transmitLUT(i,j));
   fprintf('},\n');
end
fprintf('},\n');


