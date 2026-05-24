function syllables = feather_isi(audio, fs, merge_threshold)
win_ms = 2; overlap_ratio = 0.5; min_dur_ms = 10; max_dur_ms = 200;
freq_min = 30000; freq_max = 110000;

win = round(win_ms * 1e-3 * fs);
noverlap = round(win * overlap_ratio);
nfft = win;
[~, ~, ~, Pxx] = spectrogram(audio, win, noverlap, nfft, fs, 'yaxis');

f_range = find(f >= freq_min & f <= freq_max);
spec_focus = abs(Pxx(f_range, :));
log_energy = log10(sum(spec_focus + eps, 1));
log_energy = (log_energy - min(log_energy)) / (max(log_energy) - min(log_energy));

thresh = mean(log_energy) + 0.2 * std(log_energy);
binary_env = log_energy > thresh;

if sum(binary_env) < 11
    thresh = thresh - 0.2;
    binary_env = log_energy > thresh;
end

onsets = find(diff([0 binary_env]) == 1);
offsets = find(diff([binary_env 0]) == -1);

durations = (offsets - onsets) * ((win - noverlap) / fs);
valid = durations >= min_dur_ms/1000 & durations <= max_dur_ms/1000;
syllable_start_times = t(onsets(valid));
syllable_end_times = t(offsets(valid));

i = 1;
while i < length(syllable_start_times)
    if (syllable_start_times(i+1) - syllable_end_times(i)) < merge_threshold
        syllable_end_times(i) = syllable_end_times(i+1);
        syllable_start_times(i+1) = []; syllable_end_times(i+1) = [];
    else
        i = i + 1;
    end
end

syllables.start_times = syllable_start_times;
syllables.end_times = syllable_end_times;
syllables.durations = syllable_end_times - syllable_start_times;
syllables.isi = [syllable_start_times(2:end) - syllable_end_times(1:end-1)];
end