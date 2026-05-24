clear; clc; close all;

root_paths = {
    '/Cluster1/', ...
    '/Cluster2/'
};

output_excel = '/Output/';
output_img_dir = '/Output/'; 

target_fs       = 1000;  
window_size_sec = 1;   
step_size_sec   = 0.5;   
wake_pre_sec    = 5;   
wake_post_sec   = 2;   

num_mice = 10; 
cluster_results = cell(1, 2); 

for c = 1:2
    base_path = root_paths{c};
    fprintf('\nProcessing Cluster %d (Path: %s)...\n', c, base_path);
    
    save_dir = fullfile(output_img_dir, sprintf('Cluster%d', c));
    if ~exist(save_dir, 'dir'), mkdir(save_dir); end
    
    t_axis_standard = -20 : (1/target_fs) : 20; 
    win_pts  = round(window_size_sec * target_fs);
    step_pts = round(step_size_sec * target_fs);
    
    mouse_data = []; 
    valid_mouse_count = 0;
    
    for m = 1:num_mice
        mouse_name = sprintf('Mouse_%02d', m);
        mouse_folder = fullfile(base_path, mouse_name);
        
        if ~exist(mouse_folder, 'dir'), continue; end
        
        mat_files = dir(fullfile(mouse_folder, '*.mat'));
        if isempty(mat_files)
            continue;
        end
        
        all_mouse_scores = [];
        all_mouse_labels = [];
        trial_count = 0;
        
        for f = 1:length(mat_files)
            try
                file_name = mat_files(f).name;
                S = load(fullfile(mouse_folder, file_name));
                if ~isfield(S, 'call_events') || ~isfield(S, 't_axis'), continue; end
                
                spike_times = S.t_axis(S.call_events > 0);
                bin_edges = [t_axis_standard, t_axis_standard(end) + (1/target_fs)];
                call_counts = histcounts(spike_times, bin_edges);
                
                call_smoothed = movsum(call_counts, win_pts);
                
                t_downsampled = t_axis_standard(1:step_pts:end);
                scores_trial = call_smoothed(1:step_pts:end)';
                labels_trial = (t_downsampled >= -wake_pre_sec & t_downsampled <= wake_post_sec)';
                
                all_mouse_scores = [all_mouse_scores; scores_trial];
                all_mouse_labels = [all_mouse_labels; labels_trial];
                trial_count = trial_count + 1;
                
            catch ME
                fprintf('      - Error reading/processing %s: %s\n', mat_files(f).name, ME.message);
            end
        end
        
        if length(unique(all_mouse_labels)) < 2
            continue;
        end
        
        scores_jittered = all_mouse_scores + randn(size(all_mouse_scores)) * 1e-7;
        [X, Y, ~, AUC] = perfcurve(all_mouse_labels, scores_jittered, 1);
        
        fig_mouse = figure('Visible', 'off', 'Color', 'w', 'Position', [200, 200, 500, 450]);
        hold on; grid on;
        plot([0 1], [0 1], 'k--', 'LineWidth', 1.2);
        
        if c == 1
            plot(X, Y, 'b-', 'LineWidth', 2); 
        else
            plot(X, Y, 'r-', 'LineWidth', 2); 
        end
        
        set(gca, 'Box', 'on', 'LineWidth', 1.0, 'FontSize', 11, 'XTick', 0:0.2:1.0, 'YTick', 0:0.1:1.0);        
        xlim([0 1]); ylim([0 1]); 
        xlabel('False Positive Rate'); ylabel('True Positive Rate');
        title(sprintf('Cluster %d - %s\n(n = %d trials)  AUC = %.3f', c, mouse_name, trial_count, AUC), 'Interpreter', 'none');
        
        save_name = fullfile(save_dir, sprintf('%s_AvgROC.png', mouse_name));
        exportgraphics(fig_mouse, save_name, 'Resolution', 150);
        close(fig_mouse);
        
        valid_mouse_count = valid_mouse_count + 1;
        mouse_data(valid_mouse_count).MouseName  = mouse_name;
        mouse_data(valid_mouse_count).TrialCount = trial_count;
        mouse_data(valid_mouse_count).X          = X;
        mouse_data(valid_mouse_count).Y          = Y;
        mouse_data(valid_mouse_count).AUC        = AUC;
        
        fprintf('  - [%s] Analysis complete: %d trials, AUC = %.3f\n', mouse_name, trial_count, AUC);
    end
    
    cluster_results{c} = mouse_data;
end

fprintf('\nAggregating and exporting mouse average data to Excel...\n');
if exist(output_excel, 'file'), delete(output_excel); end

auc_summary_cluster = [];
auc_summary_mouse_id = {};
auc_summary_trial_count = [];
auc_summary_values = [];

for c = 1:2
    data = cluster_results{c};
    if isempty(data), continue; end
    num_mice_valid = length(data);
    
    max_pts = 0;
    for i = 1:num_mice_valid
        max_pts = max(max_pts, length(data(i).X));
    end
    
    out_mat = NaN(max_pts, num_mice_valid * 2);
    var_names = cell(1, num_mice_valid * 2);
    
    for i = 1:num_mice_valid
        len = length(data(i).X);
        out_mat(1:len, 2*i-1) = data(i).X;
        out_mat(1:len, 2*i)   = data(i).Y;
        
        var_size_prefix = data(i).MouseName;
        var_names{2*i-1} = sprintf('%s_FPR', var_size_prefix);
        var_names{2*i}   = sprintf('%s_TPR', var_size_prefix);
        
        auc_summary_cluster(end+1, 1)     = c;
        auc_summary_mouse_id{end+1, 1}    = data(i).MouseName;
        auc_summary_trial_count(end+1, 1) = data(i).TrialCount;
        auc_summary_values(end+1, 1)      = data(i).AUC;
    end
    
    T_raw = array2table(out_mat, 'VariableNames', var_names);
    sheet_name = sprintf('C%d_Mouse_ROC_Data', c);
    writetable(T_raw, output_excel, 'Sheet', sheet_name);
end

T_auc = table(auc_summary_cluster, auc_summary_mouse_id, auc_summary_trial_count, auc_summary_values, ...
    'VariableNames', {'Cluster', 'Mouse_ID', 'Trial_Count', 'AUC'});
writetable(T_auc, output_excel, 'Sheet', 'Mice_AUC_Summary');

fprintf('All tasks completed successfully.\n');
fprintf('   1. Average ROC images saved to: %s\n', output_img_dir);
fprintf('   2. Aggregated Excel data saved to: %s\n', output_excel);