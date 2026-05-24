clear; clc; close all;

root_paths = {
    '/Data/'
};
target_folder_name = 'Extracted_Wake_Events';

fsample       = 1000;
fdsample      = 1000;
windowSize    = 1000;
shiftSize     = 500;
nfft          = 1024;

pre_wake_len  = 20;
post_wake_len = 20;
short_rem_th  = 1.0;
merge_gap_th  = 10.0;
bin_fs        = 20;
sound_energy_th = 1.0000e-05;

global_direct_wake      = 0;
global_rem_wake         = 0;
global_skipped_no_sound = 0;
global_manual_picks     = 0;

main_fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 800 400]);

for r = 1:length(root_paths)
    current_root = root_paths{r};
    fprintf('\n=======================================================\n');
    fprintf('Processing Directory [%d/%d]: %s\n', r, length(root_paths), current_root);
    
    output_main = fullfile(current_root, target_folder_name);
    if ~exist(output_main, 'dir'), mkdir(output_main); end
    
    mat_files = dir(fullfile(current_root, '*.mat'));
    
    for f = 1:length(mat_files)
        filename = fullfile(current_root, mat_files(f).name);
        [~, base_name, ~] = fileparts(mat_files(f).name);
        fprintf('\n▶ Processing File [%d/%d]: %s\n', f, length(mat_files), mat_files(f).name);
        
        try
            S = load(filename);
            if ~isfield(S, 'segments')
                fprintf('  [Skip] File does not contain "segments" variable.\n');
                continue;
            end
            segments = S.segments;
        catch ME
            fprintf('  [Error] Cannot read file: %s\n', ME.message);
            continue;
        end
        
        try
            eeg_s = vertcat(segments.EEG_raw)';
            emg_s = vertcat(segments.EMG_raw)';
            snd_p = vertcat(segments.sound_intensity)';
            fprintf('  [Step 1: Concatenation] Loaded %d segments, Total duration: %.1f s\n', length(segments), length(eeg_s)/fsample);
        catch
            fprintf('  [Error] Segment concatenation failed (possible dimension mismatch).\n');
            continue;
        end
        
        fprintf('  [Step 2: Staging] Computing sleep stages...\n');
        [s_eeg, f_eeg, t_spec, Pxx] = spectrogram(eeg_s, windowSize, shiftSize, nfft, fdsample, 'yaxis');
        
        delta = Pxx(round(1 / fdsample * 2 * size(s_eeg, 1)):round(4 / fdsample * 2 * size(s_eeg, 1)), :);
        theta = Pxx(round(6 / fdsample * 2 * size(s_eeg, 1)):round(12 / fdsample * 2 * size(s_eeg, 1)), :);
        delta_sum_t = sum(delta);
        theta_sum_t = sum(theta);
        delta_theta = theta_sum_t ./ delta_sum_t;
        
        [~, ~, ~, emgSpectrogram] = spectrogram(resample(emg_s, fdsample, fsample), windowSize, shiftSize, nfft, fdsample, 'yaxis');
        emg_sum_t = sum(emgSpectrogram(round(200 / fdsample * 2 * size(emgSpectrogram, 1)):round(500 / fdsample * 2 * size(emgSpectrogram, 1)), :), 1);
        
        thr_emg = mean(emg_sum_t) + 1 * std(emg_sum_t);
        wakei = ones(1, length(emg_sum_t));
        for wakeji = 1:length(emg_sum_t)
            if emg_sum_t(wakeji) > thr_emg
                wakei(wakeji) = 0;
            end
        end
        inds_wake = find(~wakei);
        
        thr_rem = mean(delta_theta) + 2 * std(delta_theta);
        rem_idx = find(delta_theta > thr_rem);
        commonElements = intersect(inds_wake, rem_idx);
        inds_REM = setdiff(rem_idx, commonElements);
        
        inds_NREM = 1:size(delta_sum_t, 2);
        inds_NREM([inds_REM, inds_wake]) = [];
        
        stats = zeros(1, length(delta_sum_t));
        stats(inds_REM) = 1;
        stats(inds_NREM) = 2;
        
        fprintf('    - Thresholds: EMG = %.4e | REM = %.4f\n', thr_emg, thr_rem);
        fprintf('    - Stage Counts: Wake(%d) | REM(%d) | NREM(%d)\n', length(inds_wake), length(inds_REM), length(inds_NREM));
        if length(inds_wake) < 5
            fprintf('    [Warning] Minimal Wake states identified. Extraction may fail.\n');
        end
        
        duration_eeg = length(eeg_s) / fsample;
        duration_snd = length(snd_p) / bin_fs;
        
        raw_wake_events = [];
        time_step = shiftSize / fsample;
        
        for i = 2:length(stats)
            if stats(i) == 0 && stats(i-1) ~= 0
                if stats(i-1) == 2
                    raw_wake_events = [raw_wake_events; t_spec(i), 1];
                elseif stats(i-1) == 1
                    rem_count = 0; k = i - 1;
                    while k > 0 && stats(k) == 1
                        rem_count = rem_count + 1; k = k - 1;
                    end
                    if k > 0 && stats(k) == 2
                        rem_duration = rem_count * time_step;
                        if rem_duration < short_rem_th
                            raw_wake_events = [raw_wake_events; t_spec(i), 2];
                        end
                    end
                end
            end
        end
        
        fprintf('  [Step 3: Screening] Found %d candidate transitions (NREM->Wake or NREM->shortREM->Wake)\n', size(raw_wake_events, 1));
        if isempty(raw_wake_events)
            fprintf('    [Skip] No valid candidates found in this file.\n');
            continue;
        end
        
        clusters = {}; current_cluster = [];
        for w = 1:size(raw_wake_events, 1)
            curr_t = raw_wake_events(w, 1);
            if isempty(current_cluster)
                current_cluster = raw_wake_events(w, :);
            else
                last_t = current_cluster(end, 1);
                if (curr_t - last_t) < merge_gap_th
                    current_cluster = [current_cluster; raw_wake_events(w, :)];
                else
                    clusters{end+1} = current_cluster;
                    current_cluster = raw_wake_events(w, :);
                end
            end
        end
        if ~isempty(current_cluster), clusters{end+1} = current_cluster; end
        
        fprintf('  [Step 4: Clustering] Grouped %d candidates into %d clusters (<10s gap)\n', size(raw_wake_events, 1), length(clusters));
        wake_events = []; global_t_snd = (0:length(snd_p)-1) / bin_fs;
        
        for c = 1:length(clusters)
            cluster_wakes = clusters{c};
            
            if size(cluster_wakes, 1) == 1
                wake_events = [wake_events; cluster_wakes(1, :)];
            else
                has_sound = false;
                for j = 1:size(cluster_wakes, 1)
                    cand_t = cluster_wakes(j, 1);
                    idx_in_window = global_t_snd >= (cand_t - pre_wake_len) & global_t_snd < cand_t;
                    if any(snd_p(idx_in_window) > sound_energy_th)
                        has_sound = true; break;
                    end
                end
                
                if ~has_sound
                    wake_events = [wake_events; cluster_wakes(1, :)];
                else
                    fprintf('\n    [GUI Prompt] Cluster %d detects sound proximity. Awaiting manual selection...\n', c);
                    global_manual_picks = global_manual_picks + 1;
                    
                    t_start = max(0, cluster_wakes(1, 1) - 20);
                    t_end   = min(duration_snd, cluster_wakes(end, 1) + 20);
                    idx_v   = global_t_snd >= t_start & global_t_snd <= t_end;
                    idx_stats = t_spec >= t_start & t_spec <= t_end;
                    
                    temp_fig = figure('Name', 'Manual Event Verification', 'Color', 'w', 'Position', [300, 300, 900, 400]);
                    ax_temp = axes('Parent', temp_fig);
                    
                    yyaxis(ax_temp, 'left');
                    stairs(ax_temp, t_spec(idx_stats), stats(idx_stats), 'LineWidth', 2, 'Color', [0.15 0.38 0.61]);
                    ylim(ax_temp, [-0.5 2.5]); set(ax_temp, 'YDir', 'reverse');
                    yticks(ax_temp, [0 1 2]); yticklabels(ax_temp, {'Wake', 'REM', 'NREM'});
                    ylabel(ax_temp, 'Sleep Stage', 'FontWeight', 'bold');
                    
                    yyaxis(ax_temp, 'right');
                    area(ax_temp, global_t_snd(idx_v), snd_p(idx_v), 'FaceColor', [0.85 0.33 0.1], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
                    ylabel(ax_temp, 'Sound Power', 'FontWeight', 'bold');
                    yline(ax_temp, sound_energy_th, 'k--', 'Threshold', 'LabelHorizontalAlignment', 'left');
                    
                    options = cell(size(cluster_wakes, 1) + 1, 1); colors = lines(size(cluster_wakes, 1));
                    for j = 1:size(cluster_wakes, 1)
                        cand_t = cluster_wakes(j, 1);
                        xline(ax_temp, cand_t, '-', sprintf('Option %d', j), 'Color', colors(j,:), 'LineWidth', 2.5, 'FontSize', 12, 'LabelOrientation', 'horizontal');
                        options{j} = sprintf('Option %d (t = %.1f s)', j, cand_t);
                    end
                    options{end} = 'Discard All Candidates';
                    
                    title(sprintf('File: %s | Cluster: %d | Select valid onset', base_name, c), 'Interpreter', 'none');
                    xlabel('Global Time (s)'); ylabel('Sound Power'); grid on;
                    
                    choice = menu('Select the valid physiological wake onset based on the displayed waveform:', options);
                    
                    if choice > 0 && choice <= size(cluster_wakes, 1)
                        wake_events = [wake_events; cluster_wakes(choice, :)];
                        fprintf('      -> [Action] Retained Option %d\n', choice);
                    else
                        fprintf('      -> [Action] Discarded entire cluster\n');
                    end
                    if isvalid(temp_fig), close(temp_fig); end
                end
            end
        end
        
        fprintf('  [Step 5: Export] %d verified events entering final sound check\n', size(wake_events, 1));
        saved_this_file = 0;
        
        for w = 1:size(wake_events, 1)
            t_wake = wake_events(w, 1);
            trans_type = wake_events(w, 2);
            
            t_trial_eeg = linspace(-pre_wake_len, post_wake_len, round((pre_wake_len + post_wake_len) * fsample));
            abs_t_eeg   = t_wake + t_trial_eeg;
            
            t_trial_snd = linspace(-pre_wake_len, post_wake_len, round((pre_wake_len + post_wake_len) * bin_fs));
            abs_t_snd   = t_wake + t_trial_snd;
            
            eeg_trial   = NaN(size(t_trial_eeg)); emg_trial = NaN(size(t_trial_eeg));
            stats_trial = NaN(size(t_trial_eeg)); snd_trial = zeros(size(t_trial_snd));
            
            valid_mask_eeg = (abs_t_eeg >= 0) & (abs_t_eeg <= duration_eeg);
            if any(valid_mask_eeg)
                raw_idx = round(abs_t_eeg(valid_mask_eeg) * fsample) + 1;
                eeg_trial(valid_mask_eeg) = eeg_s(min(max(raw_idx, 1), length(eeg_s)));
                emg_trial(valid_mask_eeg) = emg_s(min(max(raw_idx, 1), length(emg_s)));
                stats_trial(valid_mask_eeg) = interp1(t_spec, stats, abs_t_eeg(valid_mask_eeg), 'nearest', 'extrap');
            end
            
            valid_mask_snd = (abs_t_snd >= 0) & (abs_t_snd <= duration_snd);
            if any(valid_mask_snd)
                snd_idx = round(abs_t_snd(valid_mask_snd) * bin_fs) + 1;
                snd_trial(valid_mask_snd) = snd_p(min(max(snd_idx, 1), length(snd_p)));
            end
            
            pre_wake_mask = t_trial_snd < 0;
            pre_wake_snd  = snd_trial(pre_wake_mask);
            max_pre_snd = max(pre_wake_snd);
            
            if max_pre_snd <= sound_energy_th
                fprintf('    [Reject] Event %d discarded: Pre-wake max energy (%.2e) <= Threshold (%.2e)\n', w, max_pre_snd, sound_energy_th);
                global_skipped_no_sound = global_skipped_no_sound + 1;
                continue;
            end
            
            fprintf('    [Accept] Event %d saved: Pre-wake energy satisfies criteria.\n', w);
            saved_this_file = saved_this_file + 1;
            
            if trans_type == 1
                label_trans = "DirectWake"; global_direct_wake = global_direct_wake + 1;
            else
                label_trans = "ShortREMWake"; global_rem_wake = global_rem_wake + 1;
            end
            
            event_info.base_name  = base_name; event_info.event_num = w;
            event_info.t_wake_abs = t_wake; event_info.trans_type = label_trans;
            
            trial_data.eeg_trial   = eeg_trial; trial_data.emg_trial = emg_trial;
            trial_data.stats_trial = stats_trial; trial_data.t_trial_eeg = t_trial_eeg;
            trial_data.snd_trial   = snd_trial; trial_data.t_trial_snd = t_trial_snd;
            trial_data.event_info  = event_info;
            
            save(fullfile(output_main, sprintf('Wake_%s_%s_W%03d.mat', label_trans, base_name, w)), 'trial_data');
            
            set(0, 'CurrentFigure', main_fig); clf(main_fig); ax = axes('Parent', main_fig);
            yyaxis(ax, 'left');
            stairs(ax, t_trial_eeg, stats_trial, 'LineWidth', 2, 'Color', [0.15 0.38 0.61]);
            ylim(ax, [-0.5 2.5]); set(ax, 'YDir', 'reverse');
            yticks(ax, [0 1 2]); yticklabels(ax, {'Wake', 'REM', 'NREM'}); ylabel(ax, 'Sleep Stage');
            yyaxis(ax, 'right');
            area(ax, t_trial_snd, snd_trial, 'FaceColor', [0.85 0.33 0.1], 'FaceAlpha', 0.4, 'EdgeColor', 'none');
            ylabel(ax, 'Sound Power');
            if max(snd_trial) > 0, ylim(ax, [0, max(snd_trial) * 1.2]); end
            xline(ax, 0, '--', 'Wake Onset', 'LineWidth', 2, 'Color', [0.5 0.5 0.5]);
            xlim(ax, [-pre_wake_len, post_wake_len]); xlabel(ax, 'Time relative to Wake onset (s)');
            title(ax, sprintf('[%s] %s Event%03d', label_trans, base_name, w), 'Interpreter', 'none');
            grid(ax, 'on'); box(ax, 'on');
            
            saveas(main_fig, fullfile(output_main, sprintf('Wake_%s_%s_W%03d.png', label_trans, base_name, w)));
        end
        fprintf('  [Summary] Successfully exported %d events for this file.\n', saved_this_file);
    end
end
if exist('main_fig', 'var') && isvalid(main_fig), close(main_fig); end

fprintf('\n======================================================\n');
fprintf('======================================================\n');
total_wakes = global_direct_wake + global_rem_wake;
fprintf('▶ Total directories scanned      : %d\n', length(root_paths));
fprintf('▶ Manual GUI interventions       : %d\n', global_manual_picks);
fprintf('▶ Events dropped (no sound/noise): %d\n\n', global_skipped_no_sound);
if total_wakes > 0
    fprintf('Total Valid Events Extracted: %d\n', total_wakes);
    fprintf(' - NREM Direct Wake         : %d\n', global_direct_wake);
    fprintf(' - NREM Short-REM Wake      : %d\n', global_rem_wake);
else
    fprintf('No valid events met the extraction criteria.\n');
end
fprintf('======================================================\n\n');