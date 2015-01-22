% MY Best Response Dynamics with Fictitious Play
% Simple example with number matrix

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
    
% A = -A + max(max(A));
    
x = ones(n, 1) / n;

% f = - A * x;
% Aineq = zeros(1, n);
% bineq = 0;
% Aeq = ones(1, n);
% beq = 1;
% lb = zeros(n, 1);
% ub = ones(n, 1);
% y = linprog(f, Aineq, bineq, Aeq, beq, lb, ub);

num_cycles = 10;

double_vec = zeros(n, 2);

for cycle = 1 : num_cycles
    new_payoff = A * x;
    
%     In this way I wll never play the same pixel!
%     I am changing every time! Not suitable!
%     [~, index_max] = max(new_payoff);
%     new_BR = zeros(n, 1);
%     new_BR(index_max) = 1;
    
    new_BR = new_payoff ./ sum(new_payoff);

    old_payoff = A' * new_BR;
    sum_old_payoff = sum(old_payoff);
    
    double_vec(:, 1) = old_payoff;
    double_vec(:, 2) = x;
    
    [max_old_payoff, index_max_old_payoff] = max(old_payoff);
    [min_old_payoff, index_min_old_payoff] = min(old_payoff);
    
%     updated_probs = old_payoff;
%     updated_probs(index_max_old_payoff) = updated_probs(index_max_old_payoff) + x(index_min_old_payoff) * (max_old_payoff / (max_old_payoff + min_old_payoff));
%     updated_probs(index_min_old_payoff) = updated_probs(index_min_old_payoff) - x(index_min_old_payoff) * (max_old_payoff / (max_old_payoff + min_old_payoff));

    updated_probs = old_payoff ./ sum_old_payoff;
    

    x =  updated_probs;
    sum(x)
end

% sum(x)