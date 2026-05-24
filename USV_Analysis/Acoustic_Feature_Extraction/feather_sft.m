function [duration_true, mean_fuza, var_fuza, mean_freq, max_fre, min_fre, maxPower_Frequency, comp_std, comp_mean, inst_freq_all] = feather_sft(x, fs)

fc_low = 20000;
fc_high = 120000;
[b, a] = butter(7, [fc_low, fc_high] / (fs / 2), 'bandpass');
x_filt = filter(b, a, x);

margin = 0.000;
syllables = feather_isi(x_filt, fs, 0.05);
startidx = max(1, floor((syllables.start_times - margin) * fs));
endidx = min(floor((syllables.end_times + margin) * fs), length(x_filt));

concat_segment = [];
for i = 1:length(startidx)
    segment = x_filt(startidx(i):endidx(i));
    concat_segment = [concat_segment; segment(:)];
end

[b, a] = butter(7, [40000, 100000] / (fs / 2), 'bandpass');
audioSignal = filter(b, a, concat_segment);

N = length(audioSignal);
Y = fft(audioSignal);
P2 = abs(Y/N);
P1 = P2(1:N/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(N/2))/N;

[~, idx] = max(P2);
maxPower_Frequency = f(idx);

P1_smooth = smooth(P1, 5);
[b, a] = butter(2, 0.001, 'low');
trend = filtfilt(b, a, P1_smooth);

[peakVals, peakLocs] = findpeaks(trend);
[maxVal, maxIdx] = max(peakVals);
maxLoc = peakLocs(maxIdx);
threshold = maxVal * 0.1;

left = maxLoc;
while left > 1 && P1_smooth(left) > threshold, left = left - 1; end
right = maxLoc;
while right < length(P1_smooth) && P1_smooth(right) > threshold, right = right + 1; end

min_fre = f(left);
max_fre = f(right);

L = length(concat_segment);
nfft = 2^nextpow2(L);
Y = fft(concat_segment, nfft);
P2 = abs(Y / L);
P1 = P2(1:nfft/2+1);
P1(2:end-1) = 2*P1(2:end-1);
frequencies = fs * (0:(nfft/2)) / nfft;
P1 = P1 / sum(P1);
mean_freq = sum(frequencies .* P1');

fuza = []; complexity_std = []; complexity_mean = []; inst_freq_all = [];
syllables2 = feather_isi(x_filt, fs, 0);
startidx2 = max(1, floor(syllables2.start_times * fs));
endidx2 = min(floor(syllables2.end_times * fs), length(x_filt));

for i = 1:length(startidx2)
    segment = x_filt(startidx2(i):endidx2(i));
    [bb, aa] = butter(7, [fc_low, fc_high] / (fs / 2), 'bandpass');
    segment_filt = filter(bb, aa, segment);
    inst_freq = fre_stft(segment_filt, fs);
    inst_diff = [0; diff(inst_freq)];
    
    if length(inst_diff) > 20
        if mean(abs(inst_diff(1:20))) > 1000, inst_freq = inst_freq(20:end); end
        if mean(abs(inst_diff(end-20:end))) > 1000, inst_freq = inst_freq(1:end-20); end
    else
        if mean(abs(inst_diff(1:10))) > 2000, inst_freq = inst_freq(10:end); end
        if mean(abs(inst_diff(end-10:end))) > 2000, inst_freq = inst_freq(1:end-10); end
    end
    
    valid = (inst_freq > max_fre + 8000) | (inst_freq < min_fre - 5000);
    inst_freq = inst_freq(~valid);
    
    if mean(abs(inst_diff)) > 4000 || mean(inst_freq) > max_fre + 5000 || mean(inst_freq) < min_fre - 5000
        inst_freq = NaN;
    end
    inst_freq = smooth(inst_freq, 3);

    var_stft = var(inst_freq);
    emg_stft = sum((inst_freq - min(inst_freq)).^2);
    fuza(i) = var_stft * emg_stft / 1e17;

    curvature = diff(inst_freq, 2);
    complexity_mean(i) = mean(curvature);
    complexity_std(i) = std(curvature);
    if isnan(complexity_std(i))
        complexity_std(i) = 0; complexity_mean(i) = 0; fuza(i) = 0;
    end
    inst_freq_all = [inst_freq_all; inst_freq];
end

fuza_real = []; comp_std_real = []; comp_mean_real = [];
i = 1; n = length(fuza);
while i <= n
    c_fuza = fuza(i); c_std = complexity_std(i); c_mean = complexity_mean(i);
    while i < n && (startidx2(i+1) - endidx2(i))/fs < 0.05
        i = i + 1;
        c_fuza = c_fuza + fuza(i);
        c_std = c_std + complexity_std(i);
        c_mean = c_mean + complexity_mean(i);
    end
    fuza_real = [fuza_real, c_fuza];
    comp_std_real = [comp_std_real, c_std];
    comp_mean_real = [comp_mean_real, c_mean];
    i = i + 1;
end

comp_std = mean(comp_std_real);
comp_mean = mean(abs(comp_mean_real));
mean_fuza = mean(fuza_real(~isnan(fuza_real)));
var_fuza = std(fuza_real(~isnan(fuza_real)));
mean_freq = mean_freq;
max_fre = max(inst_freq_all);
min_fre = min(inst_freq_all);
duration_true = syllables.end_times(end) - syllables.start_times(1);
end