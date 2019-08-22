    function [zVal] = getInterpolatedZVal(vertices, point)
        zVal = [];
        inTriangles = cellfun(@(vertexSet) inpolygon(point(1),point(2),vertexSet(:,1),vertexSet(:,2)),vertices);
        triangleIndex = find(inTriangles);
        if (isempty(triangleIndex))
            return; %outside convex hull
        end
        %else calculate z values to if above or below
        vertexSet = vertices{triangleIndex};
        edge1 =  vertexSet(2,:) - vertexSet(1,:);
        edge2 =  vertexSet(3,:) - vertexSet(1,:);
        normal = cross(edge1,edge2);
        normal = normal / norm(normal);
        %set values for relevant cell surfaces
        %n dot (x - x0) = 0, solve for z coordinate
        zVal =  (point(1:2) - vertexSet(1,1:2) )*normal(1:2)'./ -normal(3) + vertexSet(1,3);
    end