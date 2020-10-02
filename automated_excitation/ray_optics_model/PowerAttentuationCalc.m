function [effectiveTransmittance] = PowerAttentuationCalc (theta, verticalDistance, meanfreepath, radiusOfCurvature)
% theta is surface normal angle, vertical distance from focal point to
% surface

%back aperture coordinates go from -1 to 1
%full width half max estimate of laser on back aperture of objective
%probably not even the same for both lasers on Gen3...
FWHM = 1.3;
sigma = FWHM / 2.36;
%NA angle in degrees of 1.2 water dipping objective
alpha = 62;

power = integral2(@(polarAngle,r) AttenuateAll(polarAngle,r).*r,0,2*pi,0,1,'method','iterated');
incidentpower = integral2(@(angle,r) gaussianValuePolar(r) .* r,0,2*pi,0,1,'method','iterated');

effectiveTransmittance = power / incidentpower;

    function [powerMatrix] = AttenuateAll(polarAngle, r)
        inputPowers = gaussianValuePolar(r);
        distancesTraveled = arrayfun(@(a,b) DistanceToCurvedPlane(a,b),r.*sin(polarAngle),r.*cos(polarAngle));
        powerMatrix = AttenuatePower(inputPowers,distancesTraveled);
    end

    function [power] = gaussianValuePolar(r)
        power = 1 / (2.*pi.*sigma.*sigma) .* exp(-(r.^2) / (2.*sigma.^2));
    end

    function [power] = AttenuatePower(inputPower, distanceTraveled)
        %see helmchen, 2005
        %P = P0 e ^ (-z/ls)
        %ls = length constant(aka mean free path)
        power = inputPower .* exp(-distanceTraveled / meanfreepath);
    end

% distance along a given vector extending from focal point to point
% (a,b) in [-1, 1] in objective back aperture space
    function [dist] = DistanceToCurvedPlane(a, b)
        % convert theta to a normal vector. direction doesnt matter since we assume
        % excitaition is radially symmetric. let vector sit in xz plane
        if theta == 90
            normal = [1 0 0];
        else
            normal = [tand(theta) 0 1];
        end
        %s is point on sphere directly above focal point
        s = normal / norm(normal) .* radiusOfCurvature;
        quadratic = [a.^2 + b.^2 + 1/ (tand(alpha).*tand(alpha)), 2.*s(1).*a+2 .* s(2).*b + 2.*(s(3) - verticalDistance) / tand(alpha),...
            dot(s,s) - radiusOfCurvature.^2 - 2.*s(3).*verticalDistance + verticalDistance.^2];
        r = roots(quadratic);
        xyz = [s(1) + a.*r, s(2) + b.*r, s(3) - verticalDistance + 1 /  tand(alpha) .*r];
        %take the correct root corresponding to higher
        if ~isreal(xyz)
            dist = 0; %this ray does interset the sphere at all
        else
            if xyz(1,3) > xyz(2,3)
                xyz = xyz(1,:);
            else
                xyz = xyz(2,:);
            end
            dist = sqrt(sum(xyz - s + [0 0 verticalDistance]).^2);
        end
    end


%     % distance along a given vector extending from focal point to point
%     % (a,b) in [-1, 1] in objective back aperture space
%     function [dist] = DistanceToNormalPlane(vertDistance, normalTheta, a, b, alpha)
%         % convert theta to a normal vector. direction doesnt matter since we assume
%         % excitaition is radially symmetric. let vector sit in xz plane
%
%         normal = [tand(normalTheta) 0 1];
%         %a and b are the x and y coordinates in back aperture
%         %find distance to normal surface
%         M = [1 0 0 -a; 0 1 0 -b; 0 0 1 -1/tand(alpha); normal 0];
%         k = [0; 0; vertDistance; 0];
%
%         %calculate the amount of power blocked out of excitaiton
%         %cone based on normal angle
%         solved = inv(M)*k;
%         dist = sqrt(sum((solved(1:3) - [0 0 vertDistance]').^2));
%     end
end