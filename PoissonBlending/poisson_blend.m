function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image

[imh, imw, nb] = size(im_s); % imh = 432, w = 768, nb = 3
% find all non-zero("1") values
[rows,cols] = find(mask_s);
k = size(rows,1); % num of "1" values

% use min and max position to form a rectangular that includes the selected region and expand 1 pixel in each direction
h_min = min(rows)-1; h_max = max(rows)+1; 
w_min = min(cols)-1; w_max = max(cols)+1;
h = h_max-h_min+1;
w = w_max-w_min+1;

V = zeros(h, w);
V(1:h*w) = 1:h*w;
% the total pixel nums of cropped rectangular 
m = h*w;

% find indices of outside neighbours of selected region
num_outside = 0;
outside_rows = zeros(1,(m-k));
outside_cols = zeros(1,(m-k));
for c = 1:w
    for r = 1:h
        orig_r = r+h_min-1;
        orig_c = c+w_min-1;
        % check if this pixel is a outside neighbour of selected region
        if(mask_s(orig_r,orig_c)==0 && ((orig_r>=2 && mask_s(orig_r-1,orig_c)==1) || (orig_r<=imh-1 && mask_s(orig_r+1,orig_c)==1) || (orig_c>=2 && mask_s(orig_r,orig_c-1)==1) || (orig_c<=imw-1 && mask_s(orig_r,orig_c+1)==1)))
           num_outside = num_outside+1; 
           outside_rows(num_outside)=r;
           outside_cols(num_outside)=c;
        end
    end
end
outside_rows = outside_rows(1:num_outside);
outside_cols = outside_cols(1:num_outside);

%TODO: consider different channel numbers
solutions = zeros(m,nb);
errors = zeros(1,nb);
img_rects = zeros(h,w,nb);

for ch = 1:nb
    %TODO: initialize counter, A (sparse matrix) and b.
    %Note: A don't have to be k��k,
    %      you can add useless variables for convenience,
    %      e.g., a total of h*w variables
    e = 0;
    b = zeros(m+num_outside, 1); % cropped rectangular pixels + outside neighbouring constraints
    num_v = (h-2)*2*3; % vertical edges 
    num_h = (w-2)*2*3; % horizontal edges
    num_in = (h-2)*(w-2)*5; % interior elements
    num_of_i = num_v+num_h+num_in+num_outside; % outside neighbouring constraints
    i = zeros(1,num_of_i);
    j = zeros(1,num_of_i);
    v = zeros(1,num_of_i);

    %TODO: fill the elements in A and b, for each pixel in the image
    % reconstruct image (cropped rectangular)
    % vertical edges 
    idx1 = V(2,1); idx2 = V(h-1,1);
    idx3 = V(2,w); idx4 = V(h-1,w);
    i(1:num_v) = [idx1:idx2,idx1:idx2,idx1:idx2,idx3:idx4,idx3:idx4,idx3:idx4];
    j(1:num_v) = [(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx3-1):(idx4-1),idx3:idx4,(idx3+1):(idx4+1)];
    v(1:num_v) = [-ones(1,h-2),2*ones(1,h-2),-ones(1,h-2),-ones(1,h-2),2*ones(1,h-2),-ones(1,h-2)];

    % horizonal edges 
    idx1 = V(1,2); idx2 = V(1,w-1);
    idx3 = V(h,2); idx4 = V(h,w-1);
    i((num_v+1):(num_v+num_h)) = [idx1:h:idx2,idx1:h:idx2,idx1:h:idx2,idx3:h:idx4,idx3:h:idx4,idx3:h:idx4];
    j((num_v+1):(num_v+num_h)) = [(idx1-h):h:(idx2-h),idx1:h:idx2,(idx1+h):h:(idx2+h),(idx3-h):h:(idx4-h),idx3:h:idx4,(idx3+h):h:(idx4+h)];
    v((num_v+1):(num_v+num_h)) = [-ones(1,w-2),2*ones(1,w-2),-ones(1,w-2),-ones(1,w-2),2*ones(1,w-2),-ones(1,w-2)];

    % interior elements
    start_idx = num_v+num_h+1;
    for c = 2:w-1
        idx1 = V(2,c); idx2 = V(h-1,c);
        delta=(idx2-idx1+1)*5;
        i(start_idx:(start_idx+delta-1)) = [idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2];
        j(start_idx:(start_idx+delta-1)) = [(idx1-h):(idx2-h),(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx1+h):(idx2+h)];
        v(start_idx:(start_idx+delta-1)) = [-ones(1,h-2),-ones(1,h-2),4*ones(1,h-2),-ones(1,h-2),-ones(1,h-2)];
        start_idx = start_idx+delta;
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
    % constraints : the outside neighbouring pixels are constant defined by target
    for idx = 1:num_outside
        r = outside_rows(idx);
        c = outside_cols(idx);
        orig_r = r+h_min-1;
        orig_c = c+w_min-1;
        e = e+1;
        i(start_idx) = e;
        j(start_idx) = V(r,c);
        v(start_idx) = 1;
        b(e) = im_t(orig_r,orig_c,ch);
        start_idx = start_idx+1;
    end
    A = sparse(i,j,v);

%     %TODO: solve the equation
%     %use "lscov" or "\", please google the matlab documents
%     size(b)
    solutions(:,ch) = A \ b;
    errors(:,ch) = sum(abs(A*solutions(:,ch)-b));

    img_rects(:,:,ch) = reshape(solutions(:,ch),[h,w]);
end
% imshow(img_rects)

%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
imgout = im_t;
ind = 1;
for c = 1:w
    for r = 1:h
        orig_r = r+h_min-1;
        orig_c = c+w_min-1;
        % replace the mask pixels
        if(ind<=k && orig_r==rows(ind) && orig_c==cols(ind)) 
            imgout(orig_r,orig_c,:) = img_rects(r,c,:);
            ind = ind+1;
        end
    end
end

disp(sum(errors))
