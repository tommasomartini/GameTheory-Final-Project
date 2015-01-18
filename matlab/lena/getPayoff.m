% funzione crea matrice di payoff data l'immagine da analizzare

function A=getPayoff(I,sigma)
I=double(I);
n=size(I,1);
m=size(I,2);
%sigma=2;    %varianza grande,cluster grande (=poco selettivo)
A=zeros(n*m);
for i=1:size(A,1)
    for j=1:size(A,2)
        if i==j
            A(i,j)=0;
        else
            a=mod(i,m);
            b=mod(j,m);
            if a==0
                a=m;
            end
            if b==0
                b=m;
            end
            p1 = I(ceil(i/m),a);
            p2 = I(ceil(j/m),b);
            A(i,j)=exp(- (abs(p1 - p2))^2 / sigma^2);
        end
    end
end

save('ALena', 'A');

end