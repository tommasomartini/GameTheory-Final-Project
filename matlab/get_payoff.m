% In input riceve l'immagine e restituisce la matrice dei payoff
function A = get_payoff(img, sigma)
    img = double(img);
    [img_height, img_width] = size(img);
    n = img_width * img_height;
    
    A = zeros(n, n);
    for i = 2 : n
        for j = 1 : i - 1
            y1 = ceil(i / img_width);
            x1 = rem(i, img_width);
            if x1 == 0
                x1 = img_width;
            end
            p1 = img(y1, x1);
            
            y2 = ceil(j / img_width);
            x2 = rem(j, img_width);
            if x2 == 0
                x2 = img_width;
            end
            p2 = img(y2, x2);
            
            metric = abs(p1 - p2);  % metric
            res = exp(- (metric)^2 / sigma^2);    % gaussian kernel of the metric
            
            A(i, j) = res;
            A(j, i) = res;
        end
    end
    
%     save('ATom', 'A');   % save the payoff matrix
end