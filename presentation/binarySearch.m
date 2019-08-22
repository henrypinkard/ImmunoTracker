    function [distance] = binarySearch(vertices, initialPoint, direction, minDistance, maxDistance)
    SEARCH_TOLERANCE = 0.1;
    
    %         fprintf('min: %d\tmax: %d\n',minDistance,maxDistance);
        halfDistance = (minDistance + maxDistance) / 2.0;
        %if the distance has been narrowed to a sufficiently small interval, return
        if (maxDistance - minDistance < SEARCH_TOLERANCE)
            distance = halfDistance;
            return
        end
        %check if point is above surface in
        searchPoint = initialPoint + direction*halfDistance;
        %         fprintf('search distance: %.0f\n',halfDistance);
        withinSurface = isWithinSurace(vertices, searchPoint);
        if (~withinSurface)
            distance = binarySearch(vertices, initialPoint, direction, minDistance, halfDistance);
        else
            distance = binarySearch(vertices, initialPoint, direction, halfDistance, maxDistance);
        end
    end