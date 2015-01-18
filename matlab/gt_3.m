% Replicator Dynamics
% 1 cluster, B&W
% Needs get_payoff.m

close all;
clear all;
clc;

%% Parameters
sigma = 150;    % standard deviation
num_cycles = 50;   % number of iterations per cluster
img_name = 'tosa.jpg'; % name of the image
thr = 80;  % percentage of the highest probabilities to keep

%%
img_col = imread(img_name);
img = rgb2gray(img_col);

figure;
imshow(img);

% size_y = 150;
% img = 190 * ones(size_y);
% img(:, end) = 200 * ones(size_y, 1);
% figure;
% imshow(uint8(img));

% load A;
A = get_payoff(img, sigma);

[img_height, img_width] = size(img);
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

% Check
% if sum(x) ~= 1
%     disp('Error! Vector x does not sum to 1:');
%     sum_x = sum(x);
%     disp(sum_x);
% end

%% Normalize the probabilities
min_prob = min(x);  % smallest probability. This will become zero
x = x - min_prob;
max_prob = max(x);
x = x ./ max_prob;

%% Display cluster
img_cluster = zeros(img_height, img_width); % in this image we only show the cluster
x_norm = x * 255;  % probability vector normalized 0-255
for i = 1 : n
    if x(i) > 1 - thr/100    % not zero prob of playing this choice
        yy = ceil(i / img_width);
        xx = rem(i, img_width);
        if xx == 0
            xx = img_width;
        end
        img_cluster(yy, xx) = x_norm(i);
        img_cluster(yy, xx) = 255;
    end
end

% figure;
% imshow(img);

figure;
imshow(img_cluster);