% Replicator Dynamics
% many clusters, color, BIG IMAGES
% Needs get_payoff_2.m

%       TODO

close all;
clear all;
clc;

%% Parameters
sigma = 100;    % standard deviation
% TODO num_cycles should be higher, as 150
num_cycles = 20;   % number of iterations per cluster
img_name = 'parrot.png'; % name of the image
thr = 95;  % percentage of the highest probabilities to keep
num_clusters = 45;   % number of clusters to find (should be automatically found!)
C = 10^(-5);    % constant to avoid zero denominators
max_num_assign_cycle = 50;   % maximum number of cycles to assign remaining pixels
max_window_size = 5;    % maximum dimension of the window to check surrounding pixels
scaling_factor = 10;

%% Main body
img_original = imread(img_name);    % acquire image
img_big_original = img_original;    % save a copy of the original image

%% Scale down the image
scaled_img = img_original(1 : scaling_factor : end, 1 : scaling_factor : end, :);
img_original = scaled_img;
img_original_double = double(img_original); % make a 'double' copy of the image

% CIELAB is perceptually linear: transform from srgb to lab
colorTransform = makecform('srgb2lab');
img_lab = applycform(img_original, colorTransform);
img = img_original;

% Show the original image
% figure;
% imshow(img_original); title('Original');

% Compute the payoff matrix
% A = get_payoff_2(img_lab, sigma);
% save('parrot_matrix', 'A');
load parrot_matrix.mat;

% Image dimensions
[img_height, img_width, ~] = size(img);
n = img_width * img_height;

flags = ones(img_height, img_width);    % '0' pixels are already inside a cluster
cluster_colors = zeros(3, num_clusters);    % contains the colors of the clusters
cluster_colors = zeros(3, 1);    % contains the colors of the clusters
cluster_color_counter = 1;  % to browse the cluster color list.
% In the end cluster_color_counter - 1 will be the number of clusters
% actually detected

x = zeros(n, 1); % (uniform) mixed strategy vector
new_x = ones(n, 1) / n;    % vector used to update x

% clustered colored image
img_mean_cluster = zeros(img_height, img_width, 3, 'uint8');

pixels_to_remove = ones(n, 1);

for cluster = 1 : num_clusters
    
    num_pixel = sum(sum(flags));    % number of non-assiged pixels
    
    %     If less than 2 pixels are left it does not make any sense to keep on
    %     clustering the image
    if num_pixel < 2
        break;
    end
    
    %     ora questo vettore contiene una uniforme tra i pixel rimasti
%     x = new_x;
%     new_x = ones(n, 1) / num_pixel;

    x = ones(n, 1) / num_pixel;
    x = x .* pixels_to_remove;

    %     Vorrei avere tutte le prob a zero, tranne quelle dei pixel non ancora
    %     in un cluster. Ma controllare quali pixel sono gia stati assegnati e
    %     lungo da fare. Tuttavia A ha le righe corrispondenti ai pixel ga
    %     presi tutte nulle. Nella moltiplicazione e' come se la prob fosse
    %     zero, perche tanto la moltiplicazione da sempre zero!
    
    for cycle = 1 : num_cycles
        y = zeros(n, 1);
        pure_payoffs = A * x;
        den = x' * A * x;
        for i = 1 : n
            %             fare cosi non falsa i risultati? se ho payoff molto piccoli
            %             rischio di avere una frazione uguale a uno! Voglio C
            %             infinnitesimo!!
            %             y(i) = x(i) * (pure_payoffs(i) + C) / (den + C);
            y(i) = x(i) * (pure_payoffs(i) + C) / (den + C);
        end
        x = y;
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
    %     sum of the probabilities of the chosen pixel. Needed to calculate the
    %     mean color of the cluster
    sum_high_probs = 0;
    
    for i = 1 : n   % for each probability in vector x
        if x(i) > 1 - thr/100    % high prob of playing this choice
            
            sum_high_probs = sum_high_probs + x(i); % sum the "high" probs
            
            %             Track back the corresponding pixel in the image
            yy = ceil(i / img_width);
            xx = rem(i, img_width);
            if xx == 0
                xx = img_width;
            end
            
            if flags(yy, xx)    % if the pixel is not assigned to a cluster
                img_cluster(yy, xx) = 255;  % color the pixel
                mask(yy, xx) = 1;   % update the mask
                
%                 row_index = (yy - 1) * img_width + xx;  % find the corresponding row of A
%                 A(row_index, :) = zeros(1, n);  % payoff 0 playing this pixel in future
                pixels_to_remove(i) = 0;

                %                 Update the avg color of the cluster
                mean_cluster_color(1) = mean_cluster_color(1) + x(i) * double(img(yy, xx, 1));
                mean_cluster_color(2) = mean_cluster_color(2) + x(i) * double(img(yy, xx, 2));
                mean_cluster_color(3) = mean_cluster_color(3) + x(i) * double(img(yy, xx, 3));
            end
        end
    end
    
    %     Update the flags: remove from the remaining pixels the ones just
    %     assigned to the current cluster
    flags = flags - mask;
    
    mean_cluster_color = uint8(mean_cluster_color / sum_high_probs);    % avg cluster color
%     cluster_colors(:, cluster_color_counter) = mean_cluster_color; % save it in the list of colors
    cluster_colors = [cluster_colors, mean_cluster_color]; % save it in the list of colors
    cluster_color_counter = cluster_color_counter + 1;  % update index to color list
    
    %     Update the clustered image
    img_mean_cluster(:, :, 1) = img_mean_cluster(:, :, 1) + mean_cluster_color(1) * uint8(mask);
    img_mean_cluster(:, :, 2) = img_mean_cluster(:, :, 2) + mean_cluster_color(2) * uint8(mask);
    img_mean_cluster(:, :, 3) = img_mean_cluster(:, :, 3) + mean_cluster_color(3) * uint8(mask);
    %     img_mean_cluster = img_mean_cluster + mean_cluster_color * uint8(mask);

%     figure
%     imshow(img_cluster)
end

cluster_colors = cluster_colors(:, 2 : end);

%% Assign the remaining pixels
% Pixels can be assigned by position or resemblance. Here we assign a pixel
% according to the membership of its neighborhood.

% unassigned_pixels = sum(sum(flags));    % if >=1 there are unassigned pixels and I have to go through the next procedure
% assign_cycle_counter = 1;
% window_size = 0;
% while unassigned_pixels && assign_cycle_counter <= max_num_assign_cycle
%     unassigned_pixels = 0;  % best case: I'll manage to assign all the remaining pixels
%     assign_cycle_counter = assign_cycle_counter + 1;    % update the cycle counter
%     window_size = min(window_size + 1, max_window_size);
%     for i = 1 : img_height  % for each pixel in the image...
%         for j = 1 : img_width
%             if flags(i, j)  % if it is not yet assigned
%                 
%                 % Assignation by RESEMBLANCE
%                 % to do...
%                 
%                 % Assignation by POSITION
%                 color_votes = zeros(1, num_clusters);   % contains the votes for each cluster color
%                 for k = max(i - window_size, 1) : min(i + window_size, img_height)  % for each pixel around...
%                     for l = max(j - window_size, 1) : min(j + window_size, img_width)
%                         curr_col = zeros(3, 1); % color of the current surrounding pixel
%                         curr_col(1) = img_mean_cluster(k, l, 1);
%                         curr_col(2) = img_mean_cluster(k, l, 2);
%                         curr_col(3) = img_mean_cluster(k, l, 3);
%                         for m = 1 : length(cluster_colors)  % for each color in the list of cluster colrs...
%                             if sum(curr_col == cluster_colors(:, m)) == 3
%                                 color_votes(m) = color_votes(m) + 1;
%                             end
%                         end
%                     end
%                 end
%                 
%                 % If there is at least one voted color among the cluster
%                 % colors...
%                 if sum(color_votes)
%                     [~, elected_colors] = max(color_votes); % list of the most voted colors
%                     elected_color = cluster_colors(:, elected_colors(1));  % one of the most voted colors
%                     img_mean_cluster(i, j, :) = elected_color;     % assigned that color to the pixel
%                     flags(i, j) = 0;    % remove the pixel from the flags
%                 else
% %                     sub_img = img(max(i - window_size, 1) : min(i + window_size, img_height), max(j - window_size, 1) : min(j + window_size, img_width), :);
% %                     figure; imshow(sub_img);
%                     unassigned_pixels = 1;  % repeat the cycle
%                 end
%             end
%         end
%     end
% end

% figure
% imshow(img_mean_cluster); title('Mean cluster');

img_mean_cluster = double(img_mean_cluster);

R = img_original_double(:, :, 1);
G = img_original_double(:, :, 2);
B = img_original_double(:, :, 3);
Rc = img_mean_cluster(:, :, 1);
Gc = img_mean_cluster(:, :, 2);
Bc = img_mean_cluster(:, :, 3);
figure(1)
subplot(221), imshow(uint8(img_mean_cluster)), axis image; title('Output image')
subplot(222), hold on 
scatter3( R(:), G(:), B(:), 80, [R(:), G(:), B(:)] / 255);
scatter3( Rc(:), Gc(:), Bc(:), 200, [Rc(:), Gc(:), Bc(:)] / 255, 'MarkerFaceColor', 'flat', 'MarkerEdgeColor', 'k');
view(35, 40);
title('Pixel distribution after clustering')
subplot(223), imshow(scaled_img), axis image; title('Original scaled image')

%% Back to the big image
% I have to pass the img as a row of triples
dataPts = reshape(img_big_original(:), size(img_big_original, 1) * size(img_big_original, 2), 3);
dataPts = dataPts';
centroids = cluster_colors;
[clustCent,point2cluster,clustMembsCell] = meanShiftCentroids(dataPts, centroids, 10, 0);

figure(222),clf,hold on
for k = 1 : size(clustCent, 2)
    myMembers = clustMembsCell{k};
    myClustCen = clustCent(:, k);
%     scatter3(dataPts(1,myMembers),dataPts(2,myMembers),dataPts(3,myMembers))
%     scatter3(dataPts(1,myMembers),dataPts(2,myMembers),dataPts(3,myMembers), 10, repmat([dataPts(1,myMembers),dataPts(2,myMembers),dataPts(3,myMembers)], size(myMembers, 2), 1), '.')
    scatter3(myClustCen(1),myClustCen(2),myClustCen(3), 80, [myClustCen(1),myClustCen(2),myClustCen(3)] / 255, 'o', 'MarkerFaceColor', 'flat', 'MarkerEdgeColor', 'k')
    view(40,35)
end
title(['Pixel space. Number of clusters: ', num2str(size(clustCent, 2))])

% img_big_cluster = zeros(size(img_big_original));
output_row = zeros(3, length(dataPts));
for i = 1 : length(dataPts)
    rel_cluster = point2cluster(i);
    output_row(:, i) = clustCent(:, rel_cluster);
end

kk = reshape(output_row', size(img_big_original, 1), size(img_big_original, 2), 3);

figure(333)
imshow(uint8(kk))

% img_big_cluster = zeros(size(img_big_original));
% img_big_cluster(1 : scaling_factor : end, 1 : scaling_factor : end, :) = img_mean_cluster;
