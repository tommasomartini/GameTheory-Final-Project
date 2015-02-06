% MY Best Response Dynamics with Fictitious Play
% many clusters, color
% needs get_payoff_2.m

close all;
clear all;
clc;

%% Parameters
img_name = 'colors.jpg'; % name of the image
sigma = 100;    % standard deviation
num_cycles = 10;   % number of iterations per cluster (should be automatically found!)
thr = 10;  % percentage of the highest probabilities to keep
redistribution_factor = 100;
num_clusters = 6;   % number of clusters to find (should be automatically found!)
scaling_factor = 30;

%% Main body

img = imread(img_name); % acquire the image...
% img = rgb2gray(img_col);    % ...and bring it in b&w

img = img(1 : scaling_factor : end, 1 : scaling_factor : end, :);

[img_height, img_width, ~] = size(img);
n = img_width * img_height; % number of pixels

% Show the original image
figure; imshow(img); title('Original');

tic
A = get_payoff_2(img, sigma); % compute the payoff matrix
toc

flags = ones(img_height, img_width);    % '0' pixels are already inside a cluster
cluster_colors = zeros(3, 1);    % contains the colors of the clusters. Needed to assign left pixels

% Probability vector. Initially set to a uniform distribution
x = ones(n, 1) / n;

img_mean_cluster = zeros(img_height, img_width, 3, 'uint8');   % clustered image

cluster_color_counter = 1;
pixels_to_remove = ones(n, 1);

for cluster = 1 : num_clusters
    
    num_pixel = sum(sum(flags));    % number of non-assiged pixels
    
    % If less than 2 pixels are left it does not make any sense to keep on
    % clustering the image
    if num_pixel < 2
        break;
    end
    
    x = ones(n, 1) / num_pixel;
    x = x .* pixels_to_remove;

    % Compute the new vector x
    can_do_better = 1;  % loop condition
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
        
        min_new_x = min(new_x);
        if min_new_x < 0
            new_x = new_x - min_new_x;
            new_x = new_x ./ sum(new_x);
        end
        
        x =  new_x;
%         sum(x)
    end
    
    %% Normalize the probabilities
    min_prob = min(x);  % smallest probability. This will become zero
    x = x - min_prob;
    max_prob = max(x);  % highest probanility. This will become one
    x = x ./ max_prob;
    
    %% Find and display the cluster
    mean_cluster_color = zeros(3, 1);     % mean color of the current cluster
    mask = zeros(img_height, img_width);
    img_cluster = zeros(img_height, img_width, 3); % in this image we show the current cluster
    % sum of the probabilities of the chosen pixel. Needed to calculate the
    % mean color of the cluster
    sum_high_probs = 0;
    for i = 1 : n   % for each probability in vector x
        if x(i) > 1 - thr/100    % high prob of playing this choice
            sum_high_probs = sum_high_probs + x(i);
            
            % Track back prob position to pixel
            yy = ceil(i / img_width);
            xx = rem(i, img_width);
            if xx == 0
                xx = img_width;
            end
            
            if flags(yy, xx)    % if the pixel is not assigned to a cluster
                img_cluster(yy, xx, :) = img(yy, xx, :);  % color the pixel
                mask(yy, xx) = 1;   % fill the mask
                
                pixels_to_remove(i) = 0;
                
                mean_cluster_color(1) = mean_cluster_color(1) + x(i) * double(img(yy, xx, 1));
                mean_cluster_color(2) = mean_cluster_color(2) + x(i) * double(img(yy, xx, 2));
                mean_cluster_color(3) = mean_cluster_color(3) + x(i) * double(img(yy, xx, 3));
            end
        end
    end
    
    flags = flags - mask;   % update flags matrix usin the mask
    
    mean_cluster_color = uint8(mean_cluster_color / sum_high_probs);    % avg color of the current cluster
    cluster_colors = [cluster_colors, mean_cluster_color];  % save this avg cluster color
    cluster_color_counter = cluster_color_counter + 1;  % I want to know how many clusters I have found so far
    img_mean_cluster(:, :, 1) = img_mean_cluster(:, :, 1) + mean_cluster_color(1) * uint8(mask);
    img_mean_cluster(:, :, 2) = img_mean_cluster(:, :, 2) + mean_cluster_color(2) * uint8(mask);
    img_mean_cluster(:, :, 3) = img_mean_cluster(:, :, 3) + mean_cluster_color(3) * uint8(mask);
    
    figure; imshow(img_cluster); title('Partial cluster');
end

cluster_colors = cluster_colors(:, 2 : end);

fprintf('Number of found clusters: %d\n', cluster_color_counter - 1);

figure; imshow(img_mean_cluster); title('Mean clusters');



