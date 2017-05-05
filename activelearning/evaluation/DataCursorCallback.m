function output_txt = myfunction(obj,event_obj)
% Display the position of the data cursor
% obj          Currently not used (empty)
% event_obj    Handle to event object
% output_txt   Data cursor text string (string or cell array of strings).
pos = get(event_obj,'Position');

% Import x and y
x = get(get(event_obj,'Target'),'XData');
y = get(get(event_obj,'Target'),'YData');

% Find index
index_x = find(x == pos(1));
index_y = find(y == pos(2));
index = intersect(index_x,index_y);

% Set output text
output_txt = {['X: ',num2str(pos(1),4)], ['Y: ',num2str(pos(2),4)], ['Index: ', num2str(index)]};
 
data = guidata(gcf);

data.xPreviewSurface.RemoveAllSurfaces; %clear old ones

indices = data.tCellImarisIndices;
% indices = data.nonTCellImarisIndices;

fprintf('Imaris index %i\n',indices(index))
func_addsurfacestosurpass(data.xImarisApp,data.surfFile,1,data.xPreviewSurface,indices(index));
%TODO: center to selection


end
