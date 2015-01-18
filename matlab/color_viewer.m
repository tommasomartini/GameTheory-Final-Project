close all;
clear all;
clc;

color = [156, 154, 68];

n = 100;
ii = zeros(n, n, 3);
for i = 1 : n
    for j = 1 : n
        ii(i, j, 1) = color(1);
        ii(i, j, 2) = color(2);
        ii(i, j, 3) = color(3);
    end
end

figure;
imshow(uint8(ii))