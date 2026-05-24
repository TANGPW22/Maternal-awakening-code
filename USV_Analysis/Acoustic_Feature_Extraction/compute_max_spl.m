function max_spl = compute_max_spl(audio, ~)
Vrms_ref = 0.07;
spl_ref = 94;
max_amp = max(abs(audio));
signal_vrms = max_amp * Vrms_ref;
max_spl = spl_ref + 20 * log10(signal_vrms / Vrms_ref);
end