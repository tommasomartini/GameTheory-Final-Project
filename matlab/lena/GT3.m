% Programma di Serena

clear all
close all
clc

I=imread('tosa.jpeg');
I=rgb2gray(I);
imshow(I)


n=size(I,1);
m=size(I,2);
x=(1/(n*m))*ones(1,n*m);
sigma=1.5;
A=getPayoff(I,sigma);         %payoff matrix

it=50;      %number off iterations

for i=1:it
    y=zeros(1,n*m);
    pi=A*x';
    piX=x*A*x';
    for j=1:n*m
       y(j)=x(j)*pi(j)/piX;
    end
    x=y;
end
for i=1:length(x)
    if x(i)~=0
        a=mod(i,m);
        if a==0
            a=m;
        end
        I(ceil(i/m),a)=255;
    end
end
figure
imshow(I)
