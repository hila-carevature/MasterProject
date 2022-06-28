% Monochromatic illumination data analysis

% data: (sample x features): features = [test_type, mean_tissue1, mean_tissue2]
% load lumavaluesa.mat
% raw_data = readtable('..\2022-06-01 Experiment 2\contrast_summary\contrast_summary_pos3.csv');

% remove heading rows : lighting_type = NaN
% raw_data(isnan(raw_data.lighting_type), :) = [];
raw_data(strcmp(raw_data.lighting_type,'lighting_type'), :) = [];

% lighting_types = unique(lumavalues.lighting_type);
[lighting_types_nb, lighting_types] = groupcounts(raw_data.lighting_type);
max_nb_datapoints = min(lighting_types_nb);
tissue_types = raw_data.Properties.VariableNames(2:end);

%% compute ANOVA test for each lighting type (compare bone-dura)
% balance number of datapoints for all lighting_types (keep nb datapoints
% same as in smallest group)

% data array (sample, tissue_type, lighting_type)
new_data = zeros(max_nb_datapoints, length(tissue_types), length(lighting_types));
p_values = [];
for i=1:length(lighting_types)

    data_temp = table2array(raw_data(strcmp(raw_data.lighting_type, lighting_types(i)), 2:end));
    % remove abundant data
    data_temp(max_nb_datapoints+1:end, :) = [];
    new_data(:, :, i) = data_temp;
%     p_values = [p_values, anova1(new_data(:, :, i), tissue_types, 'on')];
%     title(['Luminance at' lighting_types(i)])
%     ylabel('Luminance')
%     xlabel('Tissue type')
end

%% Build mean & std array 
% new_data: (sample, tissue_type, lighting_type)
[std_all, mean_all] = std(new_data);
% (tissue_type) x (illumination type)
mean_all = reshape(mean_all, length(tissue_types), length(lighting_types));
std_all = reshape(std_all, length(tissue_types), length(lighting_types));

%% Plot bone-dura contrast vs illumination type
% contrast = (dura - bone) / (dura + bone)
contrast = diff(mean_all) ./ sum(mean_all);
figure;
bar(contrast, 'grouped');
set(gca,'xticklabel', lighting_types);
ylabel("Bone-Dura Contrast")
xlabel("Illumination type")
ylim([0 0.45]);
title("Bone-Dura Contrast vs illumination - Test 2: Position 3 CLAHE")

% absolute value
figure;
bar(abs(contrast), 'grouped');
set(gca,'xticklabel', lighting_types);
ylabel("Bone-Dura Contrast Absolute")
xlabel("Illumination type")
ylim([0 0.45]);
title("Bone-Dura Contrast vs illumination - Test 2")

%% Build mean & std array for plotting errorbar Mean-luminance vs wavelength

figure;
b = bar(mean_tissue', 'grouped');
hold on
set(gca,'xticklabel', lighting_types);
% for i = 1:length(tissue_types)
%     errorbar(mean_tissue(i, :), std_tissue(i, :), '*')
%     hold on
% end

% Get the x coordinate of the bars
x = nan(length(tissue_types), length(lighting_types));
for i = 1:length(tissue_types)
    x(i,:) = b(i).XEndPoints;
end
% Plot the errorbars
errorbar(x',mean_tissue',std_tissue','k','linestyle','none');
hold off

ylabel("Luminance")
xlabel("Illumination type")
% xlim([0 1100]);
title("Luminance vs illumination type")
legend(tissue_types)


%%
mean_all = reshape(new_data, length(tissue_types), length(lighting_types));
contrast = diff(mean_all) ./ sum(mean_all);
figure;
bar(contrast, 'grouped');
set(gca,'xticklabel', lighting_types);
ylabel("Bone-Dura Contrast Absolute")
xlabel("Illumination type")
ylim([0 0.45]);
title("Bone-Dura Contrast vs illumination - Test 1")

