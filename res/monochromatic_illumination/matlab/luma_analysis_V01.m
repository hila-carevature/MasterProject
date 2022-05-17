% Monochromatic illumination data analysis

% data: (sample x features): features = [test_type, frame_nb, tissue_id, mean, std]
load lumavalues.mat

% remove heading rows : lighting_type = NaN
lumavalues(isnan(lumavalues.lighting_type), :) = [];

% lighting_types = unique(lumavalues.lighting_type);
[lighting_types_nb, lighting_types] = groupcounts(lumavalues.lighting_type);
max_nb_datapoints = min(lighting_types_nb);
tissue_types = unique(lumavalues.tissue_type);

% %% equalize imbalanced dataset -> remove abundant samples 
% % (make all lighting types have the same number of samples)
% 
% for i=1:length(lighting_types)
% %     lumavalues(, )
%     
%     
% 
% end
%% compute mean for each lighting 
% balance number of datapoints for all lighting_types (keep nb datapoints
% like smallest group
% data = [lighting_type, tissue_type, luma_mean)

% data = zeros(length(lighting_types), max_nb_datapoints, length(tissue_types));

% separate table into array [lighting_type] x [tissue_type] x [sample] x [feature] 
for i=1:length(lighting_types)
    for j=1:length(tissue_types)
%         lighting_index = lumavalues.lighting_type==lighting_types(i);
%         tissue_index = lumavalues.tissue_type==tissue_types(i);
        data_temp = lumavalues(lumavalues.lighting_type==lighting_types(i), :);
        % remove abundant data
        data_temp(max_nb_datapoints+1:end, :) = [];
        data_temp(data_temp.tissue_type == tissue_type(j), 'luma_mean')
%         data_temp = table2array(lumavalues(lumavalues.lighting_type==lighting_types(i), 'luma_mean'));

%         mean(data_temp)
%         data(i, :, j) = data_temp(1:min(lighting_types_nb));
    end
end