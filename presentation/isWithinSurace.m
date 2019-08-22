    function [within] = isWithinSurace(vertices, point)
        within = 0;
        %figure out which triangle the point is in
        zVal = getInterpolatedZVal(vertices, point);
        if isempty(zVal)
            return; %outside convex hull;
        end
        within = zVal < point(3);
    end