% MY Best Response Dynamics with Fictitious Play
% many clusters, B&W
% needs get_payoff.m


close all;
clear all;
clc;

%% Parameters
img_name = 'cat.jpg'; % name of the image
t = 5;  % initial number of individuals in the population
sigma = 150;    % standard deviation
% delta = 0.01;   % maximum distance between two probs vectors to stop the loop
num_cycles = 100;   % number of iterations per cluster (should be automatically found!)
thr = 20;  % percentage of the highest probabilities to keep
num_clusters = 3;   % number of clusters to find (should be automatically found!)

%% Main body

img_col = imread(img_name); % acquire the image...
img = rgb2gray(img_col);    % ...and bring it in b&w

% img = img(1 : 20, 1 : 20);

% img = [ 200, 200, 200, 130;
%         200, 200, 200, 65
%         200, 200, 200, 65;
%         65, 65, 65, 65];
% img = uint8(img);

% AA = 0;
% BB = 32;
% CC = 255;
% DD = 192;
% EE = 224;
% FF = 112;
% img = [ AA, AA, AA, BB, BB, BB;
%         AA, AA, CC, BB, BB, BB;
%         CC, CC, CC, CC, CC, BB;
%         DD, CC, CC, CC, EE, EE;
%         DD, CC, DD, CC, CC, EE;
%         DD, DD, DD, CC, FF, FF];
% img = uint8(img);

[img_height, img_width] = size(img);
n = img_width * img_height; % number of pixels

% img = img(1 : img_height / 2, 1 : img_width / 2);
% [img_height, img_width] = size(img);
% n = img_width * img_height; % number of pixels

% just for debugging: try to get a 0-1 image (only 2 color levels)
% for i = 1 : img_width
%     for j = 1 : img_height
%         if img(j, i) >= 127
%             img(j, i) = 220;
%         else
%             img(j, i) = 80;
%         end
%     end
% end

% Show the original image
figure; imshow(img); title('Original');

A = get_payoff(img, sigma); % compute the payoff matrix

% idea: qundo tolgo un pixel dall'immagine perche gia dentro un cluster,
% non voglio piu sceglierlo nelle prossime giocate: metto la riga
% corrsipondente della matrice a tutti -1 cosi non lo scgliero  mai e e
% probabilita si redisribuiranno tra i restanti pixel.

flags = ones(img_height, img_width);    % '0' pixels are already inside a cluster
cluster_colors = zeros(1, num_clusters);    % contains the colors of the clusters. Needed to assign left pixels

% Probability vector. Initially set to a uniform distribution
x = ones(n, 1) / n;
prev_x = zeros(n, 1);   % previous x vector
new_x = ones(n, 1) / n; % vector used to update x

img_mean_cluster = zeros(img_height, img_width, 'uint8');   % clustered image

cluster_color_counter = 1;

pixels_to_remove = ones(n, 1);

for cluster = 1 : num_clusters
    
    num_pixel = sum(sum(flags));    % number of non-assiged pixels
    
    %     If less than 2 pixels are left it does not make any sense to keep on
    %     clustering the image
    if num_pixel < 2
        break;
    end
    
    %     x = new_x;
    %     % ora questo vettore contiene una uniforme tra i pixel rimasti
    %     % now new_x contains a uniform among the remaining pixels
    %     new_x = ones(n, 1) / num_pixel;
    
    x = ones(n, 1) / num_pixel;
    x = x .* pixels_to_remove;

    %     Compute the new vector x
    can_do_better = 1;  % loop condition
    for cycle = 1 : num_cycles
        opponent_payoff = A * x;
        
        %     In this way I wll never play the same pixel!
        %     I am changing every time! Not suitable!
        max_val = max(opponent_payoff);
        opponent_BR = zeros(n, 1);
        opponent_BR(opponent_payoff == max_val) = 1;
        opponent_BR = opponent_BR ./ sum(opponent_BR);  % BR sums to one in case of multiple best responses
        
        %     new_BR = new_payoff ./ sum(new_payoff);
        
        my_payoff = A' * opponent_BR;
        %     sum_old_payoff = sum(old_payoff);
        
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
        
        ffac = 100;
        
        pos_increments = my_avg_gain(pos_gains_ind) ./ (ffac * sum_pos);
        neg_increments = my_avg_gain(neg_gains_ind) ./ (ffac * sum_neg);
        
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
        sum(x)
    end
    
    %% Normalize the probabilities
    min_prob = min(x);  % smallest probability. This will become zero
    x = x - min_prob;
    max_prob = max(x);  % highest probanility. This will become one
    x = x ./ max_prob;
    
    %% Find and display the cluster
    mean_cluster_color = 0;     % mean color of the current cluster
    mask = zeros(img_height, img_width);
    img_cluster = zeros(img_height, img_width); % in this image we show the current cluster
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
                img_cluster(yy, xx) = 255;  % color the pixel
                mask(yy, xx) = 1;   % fill the mask
                
                %                 row_index = (yy - 1) * img_width + xx;  % find the corresponding row of A
                %                 A(row_index, :) = zeros(1, n);  % payoff 0 playing this pixel in the future
                pixels_to_remove(i) = 0;
                
                mean_cluster_color = mean_cluster_color + x(i) * double(img(yy, xx));
            end
        end
    end
    
    flags = flags - mask;   % update flags matrix usin the mask
    
    mean_cluster_color = uint8(mean_cluster_color / sum_high_probs);    % avg color of the current cluster
    cluster_colors(1, cluster_color_counter) = mean_cluster_color;  % save this avg cluster color
    cluster_color_counter = cluster_color_counter + 1;  % I want to know how many clusters I have found so far
    img_mean_cluster = img_mean_cluster + mean_cluster_color * uint8(mask); % color the cluster in the mean cluster img
    
    figure; imshow(img_cluster); title('Partial cluster');
end

fprintf('Number of found clusters: %d\n', cluster_color_counter - 1);

figure; imshow(img_mean_cluster); title('Mean clusters');



