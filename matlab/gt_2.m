% Best Response Dynamics with Fictitious Play

close all;
clear all;
clc;

n = 5;  % number of choices
t = 1;  % number of individuals

% Similarity matrix, payoff matrix
A = [   0, 70, 60, 25, 15; ...
        70, 0, 90, 25, 20; ...
        60, 90, 0, 5, 25; ...
        25, 25, 5, 0, 70; ...
        15, 20, 25, 70, 0];
    
x = ones(n, 1) / n;
x = zeros(n, 1);
x(n) = 1;

% f = - A * x;
% Aineq = zeros(1, n);
% bineq = 0;
% Aeq = ones(1, n);
% beq = 1;
% lb = zeros(n, 1);
% ub = ones(n, 1);
% y = linprog(f, Aineq, bineq, Aeq, beq, lb, ub);

num_cycles = 1000;

for cycle = 1 : num_cycles
    [~, index_max] = max(A * x);
    r = zeros(n, 1);
    r(index_max) = 1;
    y = x + (r - x) / (t + 1);
    
    t = t + 1;
    x = y;
end

sum(x)