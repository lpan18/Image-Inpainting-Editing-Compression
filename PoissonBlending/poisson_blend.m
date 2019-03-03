function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image

[imh, imw, nb] = size(im_s); % imh = 432, w = 768, nb = 3
% find all non-zero("1") values
[row,col] = find(mask_s);
k = size(row,1); % num of "1" values
% use min and max to form a rectangular and expand 1 pixel in each direction
h_min = min(row)-1; h_max = max(row)+1; 
w_min = min(col)-1; w_max = max(col)+1;
h = h_max-h_min;
w = w_max-w_min;

V = zeros(h, w);
V(1:h*w) = 1:h*w;
% the total pixel nums of croped rectangular 
m = h*w;

%TODO: consider different channel numbers
solutions = zeros(m,3);
errors = zeros(1,3);
img_rects = zeros(h,w,3);

for ch = 1:nb
    %TODO: initialize counter, A (sparse matrix) and b.
    %Note: A don't have to be k��k,
    %      you can add useless variables for convenience,
    %      e.g., a total of h*w variables
    e = 0;
    b = zeros(m+m-k, 1); % m base equations + m-k constraints

    %TODO: fill the elements in A and b, for each pixel in the image
    % vertical edges 
    idx1 = V(2,1); idx2 = V(h-1,1);
    idx3 = V(2,w); idx4 = V(h-1,w);
    i = [idx1:idx2,idx1:idx2,idx1:idx2,idx3:idx4,idx3:idx4,idx3:idx4];
    j = [(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx3-1):(idx4-1),idx3:idx4,(idx3+1):(idx4+1)];
    v = [-ones(1,h-2),2*ones(1,h-2),-ones(1,h-2),-ones(1,h-2),2*ones(1,h-2),-ones(1,h-2)];

    % horizonal edges 
    idx1 = V(1,2); idx2 = V(1,w-1);
    idx3 = V(h,2); idx4 = V(h,w-1);
    i = [i, idx1:h:idx2,idx1:h:idx2,idx1:h:idx2,idx3:h:idx4,idx3:h:idx4,idx3:h:idx4];
    j = [j, (idx1-h):h:(idx2-h),idx1:h:idx2,(idx1+h):h:(idx2+h),(idx3-h):h:(idx4-h),idx3:h:idx4,(idx3+h):h:(idx4+h)];
    v = [v, -ones(1,w-2),2*ones(1,w-2),-ones(1,w-2),-ones(1,w-2),2*ones(1,w-2),-ones(1,w-2)];

    % inside elements
    for c = 2:w-1
        idx1 = V(2,c); idx2 = V(h-1,c);
        i = [i, idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2];
        j = [j, (idx1-h):(idx2-h),(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx1+h):(idx2+h)];
        v = [v, -ones(1,h-2),-ones(1,h-2),4*ones(1,h-2),-ones(1,h-2),-ones(1,h-2)];
    end
    
    % b matrix
    for c = 1:w
        for r = 1:h
            e = e+1;
            orig_r = r+h_min-1;
            orig_c = c+w_min-1;
            if((r==1&&c==1)||(r==1&&c==w)||(r==h&&c==1)||(r==h&&c==w))
                continue;
            elseif(r==1 || r==h)
                b(e) = 2*im_s(orig_r,orig_c,ch)-im_s(orig_r,orig_c-1,ch)-im_s(orig_r,orig_c+1,ch);
            elseif(c==1 || c==w)
                b(e) = 2*im_s(orig_r,orig_c,ch)-im_s(orig_r-1,orig_c,ch)-im_s(orig_r+1,orig_c,ch);
            else    
                b(e) = 4*im_s(orig_r,orig_c,ch)-im_s(orig_r,orig_c-1,ch)-im_s(orig_r,orig_c+1,ch)-im_s(orig_r-1,orig_c,ch)-im_s(orig_r+1,orig_c,ch);
            end
        end
    end
    
    %TODO: add extra constraints (if any)
    ind = 1;
    for c = 1:w
        for r = 1:h
            orig_r = r+h_min-1;
            orig_c = c+w_min-1;
            if(ind<=k && orig_r==row(ind) && orig_c==col(ind)) % inside the mask, do nothing 
                ind = ind+1;
                continue;
            else   % outside the mask, change the target to T
                e = e+1;
                i = [i, e];
                j = [j, V(r,c)];
                v = [v,1];
                b(e) = im_t(orig_r,orig_c,ch);
            end
        end
    end
    A = sparse(i,j,v);

%     %TODO: solve the equation
%     %use "lscov" or "\", please google the matlab documents
%     size(b)
    solutions(:,ch) = A \ b;
    errors(:,ch) = sum(abs(A*solutions(:,ch)-b));

    img_rects(:,:,ch) = reshape(solutions(:,ch),[h,w]);
end

imshow(img_rects)

%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
imgout = im_t;
ind = 1;
for c = 1:w
    for r = 1:h
        orig_r = r+h_min-1;
        orig_c = c+w_min-1;
        if(ind<=k && orig_r==row(ind) && orig_c==col(ind)) 
            imgout(orig_r,orig_c,:) = img_rects(r,c,:);
            ind = ind+1;
        end
    end
end

disp(sum(errors))
