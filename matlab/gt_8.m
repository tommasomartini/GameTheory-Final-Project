% Best Response Dynamics with Fictitious Play
% 1 cluster, B&W
% needs get_payoff.m
% end condition: difference between prob vectors

close all;
clear all;
clc;

%% Parameters
img_name = 'cat.jpg'; % name of the image
t = 1;  % initial number of individuals in the population
sigma = 150;    % standard deviation
delta = 0.001;   % maximum distance between two probs vectors to stop the loop
% num_cycles = 10000;   % number of iterations per cluster (should be automatically found!)
thr = 80;  % percentage of the highest probabilities to keep

%% Main body

img_col = imread(img_name); % acquire the image...
img = rgb2gray(img_col);    % ...and bring it in b&w

img = [ 2, 2, 2, 70, 70, 70;
        2, 2, 200, 70, 70, 70;
        200, 200, 200, 200, 200, 70;
        38, 200, 200, 200, 123, 123;
        38, 200, 38, 200, 200, 123;
        38, 38, 38, 200, 249, 249];
img = uint8(img);

[img_height, img_width] = size(img);
n = img_width * img_height; % number of pixels

% Show the original image
figure; imshow(img); title('Original');

A = get_payoff(img, sigma); % compute the payoff matrix

% Probability vector. Initially set to a uniform distribution
x = ones(n, 1) / n;
prev_x = zeros(n, 1);   % previous x vector

num_cycl = 0;

can_do_better = 1;  % loop condition
while can_do_better
    
    num_cycl = num_cycl + 1;
    
    [~, index_max] = max(A * x);    % position of the pure strategy which is a best response
    r = zeros(n, 1);    % best strategy r...
    r(index_max) = 1;   % ...is a pure strategy
    y = x + (r - x) / (t + 1);  % new population strategy
    
    t = t + 1;  % increment population
    prev_x = x; % update the "previous" population strategy
    x = y;      % update the population strategy
    
%     update the loop condition

% 1) norm of diff between prob vectors
%     if norm(x - prev_x) > delta     % probabilities are still strongly changing
%         can_do_better = 1;
%     else
%         can_do_better = 0;
%     end
    
    % 2) diff of highest prob
    max_diff = max(abs(x - prev_x));
    if max_diff > delta     % probabilities are still strongly changing
        can_do_better = 1;
    else
        can_do_better = 0;
    end
end

% for cycle = 1 : num_cycles
%     [~, index_max] = max(A * x);
%     r = zeros(n, 1);
%     r(index_max) = 1;
%     y = x + (r - x) / (t + 1);
%     
%     t = t + 1;
%     x = y;
% end
    
%% Normalize the probabilities
min_prob = min(x);  % smallest probability. This will become zero
x = x - min_prob;
max_prob = max(x);  % highest probability. This will become one
x = x ./ max_prob;
    
%% Display cluster
img_cluster = zeros(img_height, img_width); % in this image we show the cluster
for i = 1 : n   % for each probability
    if x(i) > 1 - thr/100    % "high" prob of playing this choice
        % Track back the image pixel (from prob vector cell to image pixel)
        yy = ceil(i / img_width);
        xx = rem(i, img_width);
        if xx == 0
            xx = img_width;
        end
        img_cluster(yy, xx) = 255;  % set the pixel to WHITE
    end
end

num_cycl

figure; imshow(img_cluster); title('Cluster');