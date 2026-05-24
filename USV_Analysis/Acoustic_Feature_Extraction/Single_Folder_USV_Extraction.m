clear; clc;

% Configuration
baseDir = 'data';
targetBatch = {'voice'};

for bb = 1:length(targetBatch)
    groupDir = fullfile(baseDir, targetBatch{bb});
    wavFiles = dir(fullfile(groupDir, '*.wav'));

    if isempty(wavFiles)
        fprintf('No wav files found in: %s\n', groupDir);
        continue;
    end

    results = [];
    for i = 1:length(wavFiles)
        wavPath = fullfile(wavFiles(i).folder, wavFiles(i).name);
        try
            [data, fs] = audioread(wavPath);
            dbmax = compute_max_spl(data, fs);
            [duration_true, mean_fuza, var_fuza, mean_fre, fremax, fremin, maxPower_Frequency, ...
             comp_std, comp_mean, ~] = feather_sft(data, fs);
            syllables = feather_isi(data, fs, 0.05);
            
            result = [i, duration_true, maxPower_Frequency, mean_fre, fremax, fremin, ...
                      mean_fuza, var_fuza, dbmax, length(syllables.durations), ...
                      mean(syllables.isi), mean(syllables.durations), comp_std, comp_mean];
            results = [results; result];
        catch ME
            fprintf('Processing error: %s\nDetails: %s\n', wavPath, ME.message);
        end
    end

    outDir = 'output';
    outFile = fullfile(outDir, [targetBatch{bb} '.xlsx']);
    writematrix(results, outFile);
    fprintf('Results saved to: %s\n', outFile);
end