% Replicator Dynamics
% Simple example with number matrix

close all;
clear all;
clc;

n = 5;

% Similarity matrix, payoff matrix
A = [   0, 70, 60, 25, 15; ...
        70, 0, 90, 25, 20; ...
        60, 90, 0, 5, 25; ...
        25, 25, 5, 0, 70; ...
        15, 20, 25, 70, 0];
    
x = ones(1, n) / n;

% x = zeros(1, n);
% x(n) = 0.8;
% x(n - 1) = 0.2;

num_cycles = 1000;

for cycle = 1 : num_cycles
    y = zeros(1, n);
    pure_payoffs = A * x';
    den = x * A * x';
    for i = 1 : n        
        y(i) = x(i) * pure_payoffs(i) / den;
    end
    x = y;
end

sum(x)