% In input riceve l'immagine e restituisce la matrice dei payoff
function A = get_payoff_2(img, sigma1)
    img = double(img);
    [img_h, img_w, ~] = size(img);
    n = img_w * img_h;
    
    A = zeros(n, n);
    
    half_win_size = 14;
    sigma2 = half_win_size;
    
    for y = 1 : img_h
        for x = 1 : img_w
            p1 = img(y, x);
            for i = max(1, y - half_win_size) : min(img_h, y + half_win_size)
                for j = max(1, x - half_win_size) : min(img_w, x + half_win_size)
                    distance = sqrt((y - i)^2 + (x - j)^2);
                    p2 = img(i, j);
                    
                    metric1 = sum((p1 - p2).^2);  % metric1: color
                    res1 = exp(- (metric1) / sigma1^2);    % gaussian kernel of metric1
                    
                    metric2 = distance^2;  % metric2: distance
                    res2 = exp(- (metric2) / sigma2^2);    % gaussian kernel of metric2
                    
                    index_y_A = img_w * (y - 1) + x;
                    index_x_A = img_w * (i - 1) + j;
                    
                    A(index_y_A, index_x_A) = res1 / 2 + res2 / 2;
                    A(index_x_A, index_y_A) = res1 / 2 + res2 / 2;
                end
            end
        end
    end
    
%     for i = 2 : n
%         for j = 1 : i - 1
%             y1 = ceil(i / img_w);
%             x1 = rem(i, img_w);
%             if x1 == 0
%                 x1 = img_w;
%             end
%             p1 = img(y1, x1, :);
%             
%             y2 = ceil(j / img_w);
%             x2 = rem(j, img_w);
%             if x2 == 0
%                 x2 = img_w;
%             end
%             p2 = img(y2, x2, :);
%             
%             metric1 = sum((p1 - p2).^2);  % metric1: color
%             res = exp(- (metric1) / sigma^2);    % gaussian kernel of the metric
%             
%             A(i, j) = res;
%             A(j, i) = res;
%         end
%     end
    
%     save('A_color', 'A');   % save the payoff matrix
end