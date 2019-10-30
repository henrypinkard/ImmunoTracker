% make sure nothing messed up in createSurfaces.m
% and that features were created correctly

myDir = '/Users/henrypinkard/Desktop/featurized_sruface_candidates';
myFiles = dir(myDir);
for k = 1:length(myFiles)
    if endsWith(myFiles(k).name, '.mat')
         %fprintf('%s\n', myFiles(k).name);
         m = matfile(sprintf('%s/%s', myDir, myFiles(k).name));
         s = m.stats;
         statsSize = size(s(1).Ids, 1);
         if statsSize == size(m, 'masks', 1)
             fprintf('Consistent: %s\n', myFiles(k).name);
         else
              fprintf('NOT Consistent: %s\n', myFiles(k).name);
         end
         try
            if sum(sum(isnan(m.features), 1))
                fprintf('Features contains NAN %s\n', myFiles(k).name); 
            end
         catch 
            fprintf('No features? %s\n', myFiles(k).name); 
         end
    end
end