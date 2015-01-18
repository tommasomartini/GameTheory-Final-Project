% Replicator Dynamics
% 1 cluster, color
% Needs get_payoff_2.m

close all;
clear all;
clc;

%% Parameters
sigma = 100;    % standard deviation
num_cycles = 10;   % number of iterations per cluster
img_name = 'cat.jpg'; % name of the image
thr = 10;  % percentage of the highest probabilities to keep

%%
img_original = imread(img_name);    % acquire image
% img_original = img_original(1 : end - 1, 1 : end - 1, :);
img_original_double = double(img_original); % make a 'double' copy of the image
% CIELAB is perceptually linear: transform from srgb to lab
% colorTransform = makecform('srgb2lab');
% img = applycform(img_original, colorTransform);
img = img_original;
figure;
imshow(img_original);

disp('Calculating payoff matrix...');

% load A_color;
A = get_payoff_2(img, sigma);

disp('Payoff matrix calculated!');

[img_height, img_width, ~] = size(img);
n = img_width * img_height;

x = ones(n, 1) / n; % (uniform) mixed strategy vector

for cycle = 1 : num_cycles
    y = zeros(n, 1);
    pure_payoffs = A * x;   
    den = x' * A * x;
    for i = 1 : n        
        y(i) = x(i) * pure_payoffs(i) / den;
    end
    x = y;
end

% % Check
% if sum(x) ~= 1
%     disp('Error! Vector x does not sum to 1:');
%     sum_x = sum(x);
%     disp(sum_x);
% end

%% Normalize the probabilities
min_prob = min(x);  % smallest probability. This will become zero
x_norm = x - min_prob;
max_prob = max(x_norm);
x_norm = x_norm ./ max_prob;

%% Identify and display the cluster
img_cluster = zeros(img_height, img_width, 3, 'uint8'); 
mean_color = zeros(3, 1);   % mean color of the cluster
for i = 1 : n
    if x_norm(i) > 1 - thr / 100;    % high prob of playing this choice        
        % track back pixel from line vector position
        yy = ceil(i / img_width);
        xx = rem(i, img_width);
        if xx == 0
            xx = img_width;
        end
        
        mean_color(1) = mean_color(1) + x(i) * img_original_double(yy, xx, 1);   % update mean color of the cluster
        mean_color(2) = mean_color(2) + x(i) * img_original_double(yy, xx, 2);
        mean_color(3) = mean_color(3) + x(i) * img_original_double(yy, xx, 3);
    end
end

mean_color_u8 = uint8(mean_color);  % convert mean color to a uint8 triple

for i = 1 : n
    if x_norm(i) > 1 - thr / 100;    % high prob of playing this choice
        
        yy = floor((i - 1) / img_width) + 1;
        xx = rem(i, img_width);
        if xx == 0
            xx = img_width;
        end
        
        img_cluster(yy, xx, :) = mean_color_u8;
%         img_color_cluster(yy, xx, :) = [255, 255, 255]';
    end
end

figure;
imshow(img_cluster);
