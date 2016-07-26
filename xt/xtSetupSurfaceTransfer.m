function [ xImarisApp, xPreviewSurface ] = xtSetupSurfaceTransfer(  )

xtIndex = 0;
javaaddpath('./ImarisLib.jar')
vImarisLib = ImarisLib;
xImarisApp = vImarisLib.GetApplication(xtIndex);
if (isempty(xImarisApp))
    error('Wrong imaris index');
end

xSurpass = xImarisApp.GetSurpassScene;
%delete old surface preview
for i = xSurpass.GetNumberOfChildren - 1 :-1: 0
   if (strcmp(char(xSurpass.GetChild(i).GetName),'Preview surface') )
      xSurpass.RemoveChild(xSurpass.GetChild(i)); 
   end
end
xPreviewSurface = xImarisApp.GetFactory.CreateSurfaces;
xPreviewSurface.SetName('Preview surface');
xSurpass.AddChild(xPreviewSurface,-1);

xPreviewSurface.RemoveAllSurfaces; %clear old ones

end

