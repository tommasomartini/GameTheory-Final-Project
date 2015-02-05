% In input riceve l'immagine e restituisce la matrice dei payoff
function A = get_payoff_int_sp(img, sigmaInt, sigmaSp)
    img = double(img);
    [img_h, img_w, ~] = size(img);
    n = img_w * img_h;
    
    A = zeros(n, n);
    for i = 2 : n
        for j = 1 : i - 1
            y1 = ceil(i / img_w);
            x1 = rem(i, img_w);
            if x1 == 0
                x1 = img_w;
            end
            p1 = img(y1, x1, :);
            
            y2 = ceil(j / img_w);
            x2 = rem(j, img_w);
            if x2 == 0
                x2 = img_w;
            end
            p2 = img(y2, x2, :);
            
            metric1 = sum((p1 - p2).^2);  % metric1: color
            metric2 = (y1 - y2)^2 + (x1 - x2)^2;   % metric2: space distance
            res = exp(- (metric1) / sigmaInt^2) * exp(- (metric2) / sigmaSp^2);    % gaussian kernel of the metric
            
            A(i, j) = res;
            A(j, i) = res;
        end
    end
    
%     save('A_color', 'A');   % save the payoff matrix
end