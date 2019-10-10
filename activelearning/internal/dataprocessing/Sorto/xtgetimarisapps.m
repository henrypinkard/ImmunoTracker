function xtImarisIDs = xtgetimarisapps(varargin)
    % GETIMARISXTAPPS Return the IDs of running Imaris XT instances
    %   Syntax
    %   ------
    %   XTGETIMARISAPPS returns the IDs of all running Imaris applications
    %   XTGETIMARISAPPS(openWindows) returns the IDs of the Imaris
    %   applications represented by the vector openWindows. The openWindows
    %   argument is a vector of numbers in the range [1:n], where n is the
    %   number of running Imaris applications instances.
    
    %% Parse the optional input.
    xtgetimarisappsParser = inputParser;
    addOptional(xtgetimarisappsParser, 'ReturnApps', [], @(arg)isvector(arg))
    
    parse(xtgetimarisappsParser, varargin{:})
    
    %% Get the install folder for program files.
    programsFolder = getenv('PROGRAMFILES');
    
    %% Find the highest Imaris version on the computer.
    imarisFolders = dir(fullfile(programsFolder, 'Bitplane\Imaris*'));
    imarisVerStrs = regexp({imarisFolders.name}', '(\d)\.(\d)\.(\d)$', 'Match', 'Once');
    imarisVerStrs = regexprep(imarisVerStrs, '\.', '');
    imarisVerNos = cellfun(@str2double, imarisVerStrs);
    imarisFolder = imarisFolders(imarisVerNos == max(imarisVerNos)).name;

    %% Create a new object from the ImarisLib class.
    if isempty(regexp(javaclasspath('-dynamic'), 'ImarisLib.jar', 'once'))
        % Construct the path string to the .jar file.
        imarislibPath = fullfile(programsFolder, 'Bitplane', imarisFolder, ...
            'XT\matlab\ImarisLib.jar');
        
        javaaddpath(imarislibPath)
    end

    xtLib = ImarisLib;

    %% Find the running Imaris instances.
    xtServer = xtLib.GetServer;    
    
    try
        xtObjectCount = xtServer.GetNumberOfObjects;
        
        xtImarisIDs = zeros(xtObjectCount, 1);
        for x = 1:xtObjectCount
            xtImarisIDs(x) = xtServer.GetObjectID(x - 1);
        end % for
    
    catch xImarisME
        xtImarisIDs = [];
        
    end % catch
    
    %% Return the desired app IDs.
    if ~isempty(xtgetimarisappsParser.Results.ReturnApps)
         xtImarisIDs = xtImarisIDs(xtgetimarisappsParser.Results.ReturnApps);
    end % if
    
end % xtgetimarisxtapps

