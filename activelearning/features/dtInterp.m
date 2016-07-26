function [ z, normal ] = dtInterp( triIndices, xCoords, yCoords, zCoords, x, y )

%first figure out if point is inside any triangle

z = nan(size(x,1),size(y,1));
for tIndex = 1:length(triIndices)
    xv = xCoords(triIndices(tIndex,:));
    yv = yCoords(triIndices(tIndex,:));
    zv = zCoords(triIndices(tIndex,:));
 
    insideTri = inpolygon(x,y,xv,yv);
    
    %find interpoated value
    v1 = [xv(1) yv(1) zv(1)] - [xv(3) yv(3) zv(3)];
    v2 = [xv(2) yv(2) zv(2)] - [xv(3) yv(3) zv(3)];
    normal = cross(v1,v2);
    %n dot (x - x0) = 0, solve for z coordinate
    zsOnPlane = arrayfun(@(x, y) dot(normal(1:2),[x y] - [xv(1) yv(1)]) ./ -normal(3) + zv(1),x,y);
    z(insideTri) = zsOnPlane(insideTri);
    fprintf('%i of %i\n',tIndex, length(triIndices));
end
end

