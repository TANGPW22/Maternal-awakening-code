clear; clc;

% Configuration
baseDir = 'data';
batch_list = {'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', ...
              'p9', 'p10', 'p11', 'p12', 'p13', 'p14'};

outFile = fullfile(baseDir, 'AllGroups_AudioFeatures.xlsx');

if exist(outFile, 'file'), delete(outFile); end

header = {'Mouse', 'AudioIndex', 'AudioFile', ...
          'DurationTrue', 'MaxPowerFrequency', 'MeanFreq', ...
          'FreMax', 'FreMin', 'MeanFuza', 'VarFuza', ...
          'DBMax', 'PitchNum', 'MeanISI', 'MeanDurationEach', 'ComplexityStd', 'ComplexityMean'};

for bb = 1:numel(batch_list)
    groupName = batch_list{bb};
    groupDir  = fullfile(baseDir, groupName);
    
    mouseFolders = dir(groupDir);
    mouseFolders = mouseFolders([mouseFolders.isdir] & ~ismember({mouseFolders.name}, {'.','..'}));
    
    sheetData = {}; 
    fprintf('=== Processing group: %s ===\n', groupName);
    
    for mf = 1:numel(mouseFolders)
        mouseName = mouseFolders(mf).name;
        audioSegDir = fullfile(groupDir, mouseName, 'audio_segments');
        wavFiles = dir(fullfile(audioSegDir, '*.wav'));
        
        if isempty(wavFiles)
            fprintf('  (No wav files found for: %s)\n', mouseName);
            continue;
        end
        
        for i = 1:numel(wavFiles)
            wavPath = fullfile(wavFiles(i).folder, wavFiles(i).name);
            try
                [data, fs] = audioread(wavPath);
                dbmax = compute_max_spl(data, fs);
                [duration_true, mean_fuza, var_fuza, mean_fre, fremax, fremin, maxPower_Frequency, ...
                 comp_std, comp_mean, ~] = feather_sft(data, fs);
                syllables = feather_isi(data, fs, 0.05);
                
                row = {mouseName, i, wavFiles(i).name, ...
                       duration_true, maxPower_Frequency, mean_fre, ...
                       fremax, fremin, mean_fuza, var_fuza, ...
                       dbmax, length(syllables.durations), mean(syllables.isi), ...
                       mean(syllables.durations), comp_std, comp_mean};
                sheetData(end+1, :) = row; 
            catch ME
                warning('Failed to process: %s. Error: %s', wavPath, ME.message);
            end
        end
    end
    
    writecell([header; sheetData], outFile, 'Sheet', groupName);
    fprintf('  -> Exported to sheet "%s" (%d entries)\n', groupName, size(sheetData,1));
end

fprintf('\nBatch processing complete. Output saved at: %s\n', outFile);