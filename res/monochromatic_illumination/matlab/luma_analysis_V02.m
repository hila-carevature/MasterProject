% Monochromatic illumination data analysis

% data: (sample x features): features = [test_type, frame_nb, tissue_id, mean, std]
load lumavaluesa.mat
raw_data = lumavaluesa;
% remove heading rows : lighting_type = NaN
raw_data(isnan(raw_data.lighting_type), :) = [];

% lighting_types = unique(lumavalues.lighting_type);
[lighting_types_nb, lighting_types] = groupcounts(raw_data.lighting_type);
max_nb_datapoints = min(lighting_types_nb);
tissue_types = raw_data.Properties.VariableNames;
tissue_types(1) = [];

%% compute mean for each lighting 
% balance number of datapoints for all lighting_types (keep nb datapoints
% same as in smallest group)

% data array (sample, tissue_type, test_type)
new_data = zeros(max_nb_datapoints, length(tissue_types), length(lighting_types));

% separate table into array [lighting_type] x [tissue_type] x [sample] x [feature] 
for i=1:length(lighting_types)
    
    data_temp = table2array(raw_data(raw_data.lighting_type==lighting_types(i), 2:end));
    % remove abundant data
    data_temp(max_nb_datapoints+1:end, :) = [];
    new_data(:, :, i) = data_temp;    
    anova1(new_data(:, :, i), tissue_types)
end

