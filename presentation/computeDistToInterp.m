    function [val, valSP] = computeDistToInterp(vertices, initialPoint, directionUnitVec,initialPointStagePositioned)
        
        SEARCH_START_DIST = 400.0;

        %binary line search to find distance to interpolation
        initialDist = SEARCH_START_DIST;
        %start with a point outside and then binary line search for the distance
        while isWithinSurace(vertices, initialPoint + directionUnitVec*initialDist)
            initialDist = initialDist*2;
        end
        val =   binarySearch(vertices, initialPoint, directionUnitVec, 0, initialDist);
        %%%
        initialDist = SEARCH_START_DIST;
        %start with a point outside and then binary line search for the distance
        while isWithinSurace(vertices, initialPointStagePositioned + directionUnitVec*initialDist)
            initialDist = initialDist*2;
        end
        valSP = binarySearch(vertices, initialPointStagePositioned, directionUnitVec, 0, initialDist);
        
    end