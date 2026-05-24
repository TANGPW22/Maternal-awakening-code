clear; clc; close all;

file_path = '/Data/'; 
manual_fs = 60; 

wake_pre_sec  = 5;    
wake_post_sec = 2;    
bin_size_sec  = 0.1;  

try
    data = readmatrix(file_path);
catch
    error('Cannot load data. Please ensure the file path points to a valid CSV/TXT file, not a directory.');
end

move_raw = data(:, 3);
pup_raw  = data(:, 4);
dam_raw  = data(:, 5);

total_len = length(dam_raw);

global_labels = zeros(total_len, 1);
wake_onsets = find(dam_raw == 2 & [0; dam_raw(1:end-1)] ~= 2);

for i = 1:length(wake_onsets)
    idx = wake_onsets(i);
    is_valid = false;
    prev_state = dam_raw(max(1, idx-1));
    
    if prev_state == 0 
        is_valid = true;
    elseif prev_state == 1 
        rem_start = idx - 1;
        while rem_start > 1 && dam_raw(rem_start) == 1
            rem_start = rem_start - 1;
        end
        if (idx - 1 - rem_start) < manual_fs && dam_raw(max(1, rem_start)) == 0
            is_valid = true;
        end
    end
    
    if is_valid
        start_idx = max(1, round(idx - wake_pre_sec * manual_fs));
        end_idx   = min(total_len, round(idx + wake_post_sec * manual_fs));
        global_labels(start_idx:end_idx) = 1;
    end
end

ds_factor = round(manual_fs * bin_size_sec);
num_bins  = floor(total_len / ds_factor);
idx_range = 1:(num_bins * ds_factor);

score_move_cont = mean(reshape(move_raw(idx_range), ds_factor, num_bins), 1, 'omitnan')';
score_pup_cont  = mean(reshape(pup_raw(idx_range), ds_factor, num_bins), 1, 'omitnan')';
labels_cont = mean(reshape(global_labels(idx_range), ds_factor, num_bins), 1)' > 0.3;

jitter = @(x) x + randn(size(x)) * 1e-8;
[X1, Y1, ~, AUC1] = perfcurve(double(labels_cont), jitter(score_move_cont), 1);
[X2, Y2, ~, AUC2] = perfcurve(double(labels_cont), jitter(score_pup_cont), 1);

t_axis_min = (1:num_bins) * bin_size_sec / 60;
full_time_lim = [0, t_axis_min(end)];

fig_move_time = figure('Color', 'w', 'Position', [100, 500, 800, 300]);
hold on; grid on;
plot(t_axis_min, score_move_cont, 'Color', [0.2 0.4 1], 'LineWidth', 1);
fill_y_move = [0, max(score_move_cont), max(score_move_cont), 0];
for i = 1:num_bins
    if labels_cont(i)
        fill([t_axis_min(i)-bin_size_sec/120, t_axis_min(i)-bin_size_sec/120, ...
              t_axis_min(i)+bin_size_sec/120, t_axis_min(i)+bin_size_sec/120], ...
              fill_y_move, 'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
ylabel('Move Score'); xlabel('Time (min)'); xlim(full_time_lim);
set(gca, 'TickDir', 'out');

fig_pup_time = figure('Color', 'w', 'Position', [100, 100, 800, 300]);
hold on; grid on;
plot(t_axis_min, score_pup_cont, 'Color', [1 0.2 0.2], 'LineWidth', 1);
for i = 1:num_bins
    if labels_cont(i)
        fill([t_axis_min(i)-bin_size_sec/120, t_axis_min(i)-bin_size_sec/120, ...
              t_axis_min(i)+bin_size_sec/120, t_axis_min(i)+bin_size_sec/120], ...
              [0 1 1 0], 'k', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    end
end
ylabel('Pupil Activity'); xlabel('Time (min)'); xlim(full_time_lim);
set(gca, 'TickDir', 'out');

fig_move_roc = figure('Color', 'w', 'Position', [950, 500, 400, 350]);
hold on; grid on;
plot([0 1], [0 1], 'k--', 'HandleVisibility', 'off', 'LineWidth', 1.2);
plot(X1, Y1, 'Color', [0.2 0.4 1], 'LineWidth', 2.5);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(['ROC: MoveScore (AUC = ', num2str(AUC1, '%.3f'), ')']);
set(gca, 'TickDir', 'out');

fig_pup_roc = figure('Color', 'w', 'Position', [950, 100, 400, 350]);
hold on; grid on;
plot([0 1], [0 1], 'k--', 'HandleVisibility', 'off', 'LineWidth', 1.2);
plot(X2, Y2, 'Color', [1 0.2 0.2], 'LineWidth', 2.5);
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(['ROC: Pupil State (AUC = ', num2str(AUC2, '%.3f'), ')']);
set(gca, 'TickDir', 'out');

base_dir = './Output/ROC_Results_Continuous';
if ~exist(base_dir, 'dir'), mkdir(base_dir); end
excel_filename = fullfile(base_dir, 'Continuous_ROC_Analysis_Results.xlsx');

T_Timeline = table(t_axis_min(:), score_move_cont(:), score_pup_cont(:), double(labels_cont(:)), ...
    'VariableNames', {'Time_min', 'MoveScore_Binned', 'PupilState_Binned', 'Wake_Label'});
writetable(T_Timeline, excel_filename, 'Sheet', 'Timeline_Data');

T_Move_ROC = table(X1(:), Y1(:), 'VariableNames', {'FPR_X', 'TPR_Y'});
T_Pup_ROC = table(X2(:), Y2(:), 'VariableNames', {'FPR_X', 'TPR_Y'});
writetable(T_Move_ROC, excel_filename, 'Sheet', 'MoveScore_ROC_Points');
writetable(T_Pup_ROC, excel_filename, 'Sheet', 'PupilState_ROC_Points');