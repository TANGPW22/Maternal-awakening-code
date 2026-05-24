clear; clc; close all;

input_dir = 'C:\Users\imoge\Desktop\57julei\Clustering_Analysis_Only\聚类样本';       
output_root = 'C:\Users\imoge\Desktop\57julei\output'; 
if ~exist(output_root, 'dir'), mkdir(output_root); end                              

bin_size_ms  = 100;                                                                 
smooth_span  = 15;                                                                  
classify_win = [-10, 0];                                                            
full_win     = [-20, 20];                                                           

my_colors = [0.4 0.65 0.95; 0.95 0.65 0.4; 0.4 0.8 0.5; 0.7 0.5 0.9];               

fprintf('[Step 1] Reading full sample data and extracting density features...\n');                        
files = dir(fullfile(input_dir, '*.mat'));                                          
raw_matrix_full = []; valid_filepaths = {}; valid_filenames = {}; raw_spike_times = {}; 
edges = linspace(full_win(1), full_win(2), ((full_win(2) - full_win(1)) * 1000 / bin_size_ms) + 1); 
t_binned = edges(1:end-1) + diff(edges)/2;                                          

for i = 1:length(files)                                                             
    path = fullfile(input_dir, files(i).name);                                      
    try                                                                             
        S = load(path);                                                             
        if ~isfield(S, 'call_events'), continue; end                                
        spike_times = S.t_axis(S.call_events == 1);                                 
        counts = histcounts(spike_times, edges);                                    
        w_smooth = smoothdata(counts, 'gaussian', smooth_span);                     
        raw_matrix_full = [raw_matrix_full; w_smooth];                              
        raw_spike_times{end+1} = spike_times(:);                                    
        valid_filepaths{end+1} = path;                                              
        valid_filenames{end+1} = files(i).name;                                     
    catch ME                                                                          
    end                                                                             
end                                                                                 
if isempty(raw_matrix_full), error('Data reading failed. Please check the input path.'); end                       

fprintf('[Step 2] Identifying optimal cluster number using the Elbow Method...\n');                          
idx_classify = (t_binned >= classify_win(1) & t_binned <= classify_win(2));

normalize_flag = true; 

if normalize_flag
    classify_matrix = zscore(raw_matrix_full(:, idx_classify), 0, 2);
    classify_matrix(isnan(classify_matrix)) = 0; 
else
    classify_matrix = raw_matrix_full(:, idx_classify);
end

classify_matrix = raw_matrix_full(:, idx_classify);                                 
[~, scores_all] = pca(classify_matrix);                                             
features = scores_all(:, 1:3);                                                      
K_range = 1:10;                                                                     
sse = zeros(size(K_range));                                                         

for i = 1:length(K_range)                                                           
    [~, ~, sumd] = kmeans(features, K_range(i), 'Replicates', 50);                  
    sse(i) = sum(sumd);                                                             
end                                                                                 

pts = [K_range', sse'];                                                             
v_line = pts(end,:) - pts(1,:);                                                     
v_line = v_line / norm(v_line);                                                     
v_to_start = pts - pts(1,:);                                                        
dist = abs(v_to_start(:,1)*v_line(2) - v_to_start(:,2)*v_line(1));                  
[~, best_k_idx] = max(dist);                                                        
num_clusters = K_range(best_k_idx);                                                 
if num_clusters < 2, num_clusters = 2; end                                          
fprintf('    Optimal K = %d\n', num_clusters);                

fig_elbow = figure('Color', 'w', 'Name', 'Optimal K Selection');                    
plot(K_range, sse, '-ko', 'LineWidth', 1.5, 'MarkerFaceColor', 'r'); grid on;       
title(['Elbow Method: Best K = ', num2str(num_clusters)]); xlabel('Number of Clusters (K)'); ylabel('Total SSE'); 

fprintf('[Step 3] Executing K-means clustering and archiving files...\n');                         
[cluster_labels, ~] = kmeans(features, num_clusters, 'Replicates', 100);            

avg_density = mean(classify_matrix, 2);                                             
c_dens = arrayfun(@(k) mean(avg_density(cluster_labels == k)), 1:num_clusters);     
[~, sorted_idx] = sort(c_dens, 'descend');                                          
new_labels = zeros(size(cluster_labels));                                           
for k = 1:num_clusters, new_labels(cluster_labels == sorted_idx(k)) = k; end        
cluster_labels = new_labels;                                                        

for k = 1:num_clusters                                                              
    c_path = fullfile(output_root, sprintf('Cluster%d', k));                        
    if ~exist(c_path, 'dir'), mkdir(c_path); end                                    
    idx_k = find(cluster_labels == k);                                              
    for j = 1:length(idx_k)                                                         
        copyfile(valid_filepaths{idx_k(j)}, c_path);                                
    end                                                                             
end                                                                                 

fig_3d = figure('Color', 'k', 'Position', [100, 100, 900, 700]); hold on;           
camproj('perspective');                                                             
x_lim = [min(features(:,1)), max(features(:,1))]*1.2;                               
y_lim = [min(features(:,2)), max(features(:,2))]*1.2;                               
z_lim = [min(features(:,3)), max(features(:,3))]*1.2;                               
h_scatters = gobjects(num_clusters, 1);                                             
for k = 1:num_clusters                                                              
    idx = (cluster_labels == k);                                                    
    h_scatters(k) = scatter3(features(idx,1), features(idx,2), features(idx,3), 70, ... 
                             my_colors(k,:), 'filled', 'MarkerEdgeColor', 'w');     
end                                                                                 

if num_clusters == 2                                                                
    grid_res = 100;                                                                 
    [GX, GY, GZ] = meshgrid(linspace(x_lim(1), x_lim(2), grid_res), linspace(y_lim(1), y_lim(2), grid_res), linspace(z_lim(1), z_lim(2), grid_res)); 
    c1 = mean(features(cluster_labels==1, :), 1); c2 = mean(features(cluster_labels==2, :), 1); 
    D_diff = reshape(sqrt(sum(([GX(:), GY(:), GZ(:)]-c1).^2, 2)) - sqrt(sum(([GX(:), GY(:), GZ(:)]-c2).^2, 2)), size(GX)); 
    p_plane = isosurface(GX, GY, GZ, D_diff, 0);                                    
    if ~isempty(p_plane.vertices)                                                   
        patch(p_plane, 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.15, 'FaceLighting', 'gouraud'); 
    end                                                                             
end                                                                                 
view(135, 25); lighting gouraud; camlight headlight; material shiny;                
set(gca, 'Color', 'k', 'XColor', 'w', 'YColor', 'w', 'ZColor', 'w', 'GridAlpha', 0.05); 
xlabel('PC1'); ylabel('PC2'); zlabel('PC3'); grid on;                               
title(sprintf('Final PCA Space (K=%d)', num_clusters), 'Color', 'w');               

fprintf('[Step 4] Computing PSTH and exporting CSV data...\n');                           
bin_width_psth = 2.0;                                                               
edges_psth = full_win(1) : bin_width_psth : full_win(2);                            
t_centers = (edges_psth(1:end-1) + bin_width_psth/2)';                              
psth_save_dir = fullfile(output_root, 'PSTH_Plots'); if ~exist(psth_save_dir, 'dir'), mkdir(psth_save_dir); end 

t_export = t_centers;                                                               
export_cols = {t_export};                                                           
var_names = {'Time_Relative_to_Wake_s'};                                            
max_hz = 0; psth_mats = cell(num_clusters, 1); thresh_vals = zeros(num_clusters, 1); 

for k = 1:num_clusters                                                              
    idx = (cluster_labels == k); n = sum(idx);                                      
    if n == 0, continue; end                                                        
    hz = histcounts(cell2mat(raw_spike_times(idx)'), edges_psth) / (n * bin_width_psth); 
    psth_mats{k} = hz(:);                                                           
    
    b_idx = (t_centers >= -20 & t_centers <= -10);                                  
    thresh_vals(k) = mean(hz(b_idx)) + 3*std(hz(b_idx));                            
    max_hz = max([max_hz, max(hz), thresh_vals(k)]);                                
    
    export_cols{end+1} = psth_mats{k};                                              
    export_cols{end+1} = ones(size(t_export)) * thresh_vals(k);                     
    var_names{end+1} = sprintf('Cluster%d_Frequency_Hz', k);                        
    var_names{end+1} = sprintf('Cluster%d_Threshold_3SD', k);                       
end                                                                                 

for k = 1:num_clusters                                                              
    if isempty(psth_mats{k}), continue; end                                         
    fig_psth = figure('Color', 'w', 'Position', [100, 100, 500, 350]); hold on;     
    bar(t_centers, psth_mats{k}, 1, 'FaceColor', my_colors(k,:), 'EdgeColor', 'w', 'FaceAlpha', 0.8); 
    yline(thresh_vals(k), '--', 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);          
    xline(0, '--r', 'Wake');                                                        
    xlim(full_win); ylim([0, max_hz * 1.2]); grid on;                               
    title(sprintf('Cluster %d PSTH (n=%d)', k, sum(cluster_labels==k)));            
    saveas(fig_psth, fullfile(psth_save_dir, sprintf('PSTH_Cluster_%d.png', k)));   
end                                                                                 

psth_table = table(export_cols{:}, 'VariableNames', var_names);                     
timestamp = datestr(now, 'yyyymmdd_HHMMSS');                                        
csv_filename = sprintf('Clustering_PSTH_Data_K%d_%s.csv', num_clusters, timestamp); 
writetable(psth_table, fullfile(output_root, csv_filename));                        

fprintf('Task completed successfully.\n');                                                      
fprintf('    - Optimal K identified and %d samples clustered and archived.\n', length(cluster_labels));   
fprintf('    - Data exported to: %s\n', csv_filename);