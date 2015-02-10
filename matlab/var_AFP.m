% Replicator Dynamics
% many clusters, color, BIG IMAGES
% Needs get_payoff_2.m

%       TODO

close all;
clear all;
clc;

%% Parameters
sigma_intensity = 100;    % standard deviation
sigma_space = 30;
sigma_intensity = 50;    % standard deviation
sigma_space = 70;
num_cycles_vector = 50 : 1 : 50;
img_name = 'parrot.png'; % name of the image
thr = 10;  % percentage of the highest probabilities to keep
redistribution_factor = 100;
num_clusters = 100;   % number of clusters to find (should be automatically found!)
scaling_factor = 10;
gaussian_window = 20;
gaussian_variance = 5;

%% Acquire image
img_original = imread(img_name);    % acquire image
img_big_original = img_original;    % save a copy of the original image

%% Scale down the image
gaussian_filter = fspecial('gaussian', [gaussian_window, gaussian_window], gaussian_variance);
img_padded = padarray(img_original, gaussian_window, 'symmetric');
img_filtered = imfilter(img_padded, gaussian_filter, 'same');
img_filtered = img_filtered(1 + gaussian_window : end - gaussian_window, 1 + gaussian_window : end - gaussian_window, :);
scaled_img = img_filtered(1 : scaling_factor : end, 1 : scaling_factor : end, :);
img_original = scaled_img;
img_original_double = double(img_original); % make a 'double' copy of the image

%% Convert the color space
% CIELAB is perceptually linear: transform from srgb to lab
colorTransform = makecform('srgb2lab');
img_lab = applycform(img_original, colorTransform);
img = img_original;

% Show the original image
% figure;
% imshow(img_original); title('Original');

%% Compute the payoff matrix
A = get_payoff_int_sp(img_lab, sigma_intensity, sigma_space);
% save('parrot', 'A');
% load parrot_matrix.mat;

%% Compute the globl variance
row_img = reshape(img, size(img, 1) * size(img, 2), 3);
mean_col = (mean(row_img))';
mean_col_rep = repmat(mean_col, 1, size(row_img, 1));
global_variance = mean(sum((double(row_img') - mean_col_rep).^2));

%% Computations valid for each cycle
% Image dimensions
[img_height, img_width, ~] = size(img);
n = img_width * img_height;

% In these vectors each element is relative to a num_cycle
intra_cluster_variances = [];
inter_cluster_variances = [];
inter_cluster_variances = [];

%% FOR: number of cycles
for num_it = 1 : length(num_cycles_vector)
    num_cycles = num_cycles_vector(num_it);
    
    clusters_variances = []; % for this num_cycle, the variance for each cluster
    flags = ones(img_height, img_width);    % '0' pixels are already inside a cluster
    cluster_colors = [];    % contains the colors of the clusters
    img_mean_cluster = zeros(img_height, img_width, 3, 'uint8'); % clustered colored image
    pixels_to_remove = ones(n, 1);

%     x = zeros(n, 1); % (uniform) mixed strategy vector
%     new_x = ones(n, 1) / n;    % vector used to update x
    
%% FOR: one cylce for each cluster to find
    for cluster = 1 : num_clusters
        
        num_pixel = sum(sum(flags));    % number of non-assiged pixels
        
        % If less than 5 pixels are left it does not make any sense to keep on
        % clustering the image
        if num_pixel < 5
            break;
        end
        
        x = ones(n, 1) / num_pixel;
        x = x .* pixels_to_remove;
        
        %% FOR: iterations with num_cycle
        for cycle = 1 : num_cycles
            opponent_payoff = A * x;
            
            max_val = max(opponent_payoff);
            opponent_BR = zeros(n, 1);
            opponent_BR(opponent_payoff == max_val) = 1;
            opponent_BR = opponent_BR ./ sum(opponent_BR);  % BR sums to one in case of multiple best responses
            
            my_payoff = A' * opponent_BR;
            
            my_avg_payoff = my_payoff' * x;
            my_avg_gain = my_payoff - my_avg_payoff;
            
            % In order not to fixing choosing the same exact pixel, the diagonal of
            % the payoff matrix is set to zero. Thus, if I play the same pixel of
            % my opponent, we both get zero, even though it belongs to the same
            % cluster. I don't want it to be my best response, but I want it to be
            % a feasible choice! Since the matrix is non zero and the only zero
            % values are in the diagonal, the only way I can get a zero payoff is
            % playing the same pixel. Therefore playing the same pixel leads me to
            % the lower gain. I make it positive in order to count it as a good
            % pixel.
            %
            % This case is lucky! But what if the max positive is lower than the
            % min negative?? The same pixel becomes the best response.
            %
            % SOLUTION: I don't increse its probabilty! It is likely already quite
            % high
            
            min_gain_pos = find(my_avg_gain == -my_avg_payoff);
            my_avg_gain(min_gain_pos) = 0;
            
            pos_gains_ind = find(my_avg_gain > 0);
            neg_gains_ind = find(my_avg_gain < 0);
            
            sum_pos = sum(my_avg_gain(pos_gains_ind));
            sum_neg = -sum(my_avg_gain(neg_gains_ind));
            
            pos_increments = my_avg_gain(pos_gains_ind) ./ (redistribution_factor * sum_pos);
            neg_increments = my_avg_gain(neg_gains_ind) ./ (redistribution_factor * sum_neg);
            
            %     sum(pos_increments) + sum(neg_increments)
            
            new_x = x;
            new_x(pos_gains_ind) = new_x(pos_gains_ind) + pos_increments;
            new_x(neg_gains_ind) = new_x(neg_gains_ind) + neg_increments;
            
%             min_new_x = min(new_x);
%             if min_new_x < 0
%                 new_x = new_x - min_new_x;
%                 new_x = new_x ./ sum(new_x);
%             end
            
            min_new_x = min(new_x);
            new_x = new_x - min_new_x;
            max_new_x = max(new_x);
            
            
            x =  new_x;
        end
        
        %% Normalize the probabilities
        min_prob = min(x);  % smallest probability. This will become zero
        x = x - min_prob;
        max_prob = max(x);  % highest probability. This will become 1
        x = x ./ max_prob;
        
        %% Find and display the cluster
        mean_cluster_color = zeros(3, 1);       % mean color of the current cluster
        mask = zeros(img_height, img_width);    % mask of the current cluster
        img_cluster = zeros(img_height, img_width); % b&w image. Cluster in white on black
        % sum of the probabilities of the chosen pixel. Needed to calculate the
        % mean color of the cluster
        sum_high_probs = 0;
        
        %% FOR: iterations over the x vector to find the cluster
        for i = 1 : n   % for each probability in vector x
            if x(i) > 1 - thr/100    % high prob of playing this choice
                
                sum_high_probs = sum_high_probs + x(i); % sum the "high" probs
                
                % Track back the corresponding pixel in the image
                yy = ceil(i / img_width);
                xx = rem(i, img_width);
                if xx == 0
                    xx = img_width;
                end
                
                if flags(yy, xx)    % if the pixel is not assigned to a cluster
                    img_cluster(yy, xx) = 255;  % color the pixel
                    mask(yy, xx) = 1;   % update the mask
                    pixels_to_remove(i) = 0;
                    
                    % Update the avg color of the cluster
                    mean_cluster_color(1) = mean_cluster_color(1) + x(i) * double(img(yy, xx, 1));
                    mean_cluster_color(2) = mean_cluster_color(2) + x(i) * double(img(yy, xx, 2));
                    mean_cluster_color(3) = mean_cluster_color(3) + x(i) * double(img(yy, xx, 3));
                end
            end
        end
        
        % Update the flags: remove from the remaining pixels the ones just
        % assigned to the current cluster
        flags = flags - mask;
        
        mean_cluster_color = uint8(mean_cluster_color / sum_high_probs);    % avg cluster color
        cluster_colors = [cluster_colors, mean_cluster_color]; % save it in the list of colors
       
        %% Variance of the cluster
        rep_mask = repmat(mask, 1, 1, 3);
        clust_elems = img_original(rep_mask > 0);
        pixel_values = reshape(clust_elems, sum(sum(mask)), 3);
        pixel_values = double(pixel_values');
        rep_clust_color = repmat(double(mean_cluster_color), 1, size(pixel_values, 2));
        cluster_variance = mean(sum((pixel_values - rep_clust_color).^2));
        clusters_variances = [clusters_variances, cluster_variance];
                
        % Update the clustered image
        img_mean_cluster(:, :, 1) = img_mean_cluster(:, :, 1) + mean_cluster_color(1) * uint8(mask);
        img_mean_cluster(:, :, 2) = img_mean_cluster(:, :, 2) + mean_cluster_color(2) * uint8(mask);
        img_mean_cluster(:, :, 3) = img_mean_cluster(:, :, 3) + mean_cluster_color(3) * uint8(mask);
        
        %     figure
        %     imshow(img_cluster)
    end
        
    inter_cluster_variances = [inter_cluster_variances, mean(sum((repmat(mean_col, 1, size(cluster_colors, 2)) - double(cluster_colors)).^2))];
    intra_cluster_variances = [intra_cluster_variances, mean(clusters_variances)];
    
%     figure
%     imshow(img_mean_cluster)
    
end

figure
plot(num_cycles_vector, sqrt(intra_cluster_variances))
title('intra')

figure
plot(num_cycles_vector, sqrt(inter_cluster_variances))
title('inter')


