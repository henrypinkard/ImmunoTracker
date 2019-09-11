clear
divisions = 16; %must be even<--currently hard coded into arduino code
phaseOffset = 5; %degrees<--here's your start offset: (valid between 0 and 90)
frequency = 7923.86; %Hz<--leave this fixed
timeResolution = 4; %us<--change this as you'd like


period = 1/(frequency)*1e6; %microseconds
%phaseOffset = phaseOffset/360*period;

t = linspace(0,period,1000);
position = -(cosd(t(1:500)*360/period)-1)/2;

plot(t,[position,-position+1])

hold on

posdelay = cosd(0)-cosd(phaseOffset);

pospoints = linspace(posdelay,1-posdelay,divisions/2+1);
timepoints = zeros(size(pospoints));
for a = 1:divisions/2+1
    timepoints(a) = t(find(position>=pospoints(a),1,'first'));
end


%plot(timepoints,pospoints,'ro');

%timedelay2 = t(500)-timepoints(end);
%timedelays = timepoints - timepoints(end);
finaltimes = ones(1,divisions+2)*t(500);
finaltimes(1:divisions/2+1) = timepoints;
finaltimes(divisions/2+2:end) = timepoints+period/2;
%finaltimes(divisions/2+2:end) = t(500)+fliplr(diff([timepoints,period/2]));
finalpos = [pospoints,-pospoints+1];


plot(finaltimes,finalpos,'ro');

roundedtimes = round([0,finaltimes,period]/timeResolution)*timeResolution;

plot(roundedtimes,[0,finalpos,0],'bo');

timeinputs = roundedtimes/timeResolution;

finalval = diff(timeinputs);
finalval = finalval(1:ceil(numel(finalval)/2));
finalval = [finalval,fliplr(finalval)];









