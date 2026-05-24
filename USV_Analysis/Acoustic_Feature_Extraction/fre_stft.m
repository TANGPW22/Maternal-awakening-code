function inst_freq_stft = fre_stft(y, fs)
win = round(fs * 0.001);
overlap = win / 2;
nfft = win;
[s, f, ~] = spectrogram(y, win, overlap, nfft, fs);
[~, idx] = max(abs(s), [], 1);
inst_freq_stft = f(idx);
end