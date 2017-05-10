clear
%change to current dir
cd(fileparts(mfilename('fullpath')))
%make colored surface
[x y z] = sphere(500);
ind = x > 0 & y > 0 & z > 0;
width = sum(ind(end-1,:));
x = reshape(x(ind),[],width);
y = reshape(y(ind),[],width);
z = reshape(z(ind),[],width);

depth = 256;
 colorIndex = x *depth...
     + y*depth*depth - mod(y*depth*depth, depth) + ...
     + z*depth*depth*depth - mod(z*depth*depth*depth, depth*depth);
surface(x,y,z, colorIndex,'CDataMapping','direct','EdgeColor','none','LineStyle','none')

map = [repmat((1:depth)',depth*depth,1) ./ depth...
    repmat((1:depth*depth)',depth,1) ./ (depth*depth)...
    repmat((1:depth*depth*depth)',1,1) ./ (depth*depth*depth) ];
colormap(map)
alpha(0.5)
grid on
view(85,34) 
% print('Hypersphere figure','-depsc')


