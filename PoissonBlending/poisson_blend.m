function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image

[imh, imw, nb] = size(im_s); % imh = 567, imw = 988, nb = 3

% indices=find(mask_s);
% for i=1:size(mask_s)[0]
%     for j=1:size(mask_s)[1]
%         if(mask_s(i,j) == 1)
%             disp(i);
%             disp(j);
%         end
%     end
% end
        

V = zeros(imh, imw);
V(1:imh*imw) = 1:imh*imw;

m = imh*imw;
%TODO: consider different channel numbers
solutions = zeros(m,3);
errors = zeros(1,3);

for ch = 1:nb
    %TODO: initialize counter, A (sparse matrix) and b.
    %Note: A don't have to be k��k,
    %      you can add useless variables for convenience,
    %      e.g., a total of imh*imw variables
    e = 0;
    b = zeros(m, 1);

    %TODO: fill the elements in A and b, for each pixel in the image
    % vertical edges 
    idx1 = V(2,1); idx2 = V(imh-1,1);
    idx3 = V(2,imw); idx4 = V(imh-1,imw);
    i = [idx1:idx2,idx1:idx2,idx1:idx2,idx3:idx4,idx3:idx4,idx3:idx4];
    j = [(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx3-1):(idx4-1),idx3:idx4,(idx3+1):(idx4+1)];
    v = [-ones(1,imh-2),2*ones(1,imh-2),-ones(1,imh-2),-ones(1,imh-2),2*ones(1,imh-2),-ones(1,imh-2)];

    % horizonal edges 
    idx1 = V(1,2); idx2 = V(1,imw-1);
    idx3 = V(imh,2); idx4 = V(imh,imw-1);
    i = [i, idx1:imh:idx2,idx1:imh:idx2,idx1:imh:idx2,idx3:imh:idx4,idx3:imh:idx4,idx3:imh:idx4];
    j = [j, (idx1-imh):imh:(idx2-imh),idx1:imh:idx2,(idx1+imh):imh:(idx2+imh),(idx3-imh):imh:(idx4-imh),idx3:imh:idx4,(idx3+imh):imh:(idx4+imh)];
    v = [v, -ones(1,imw-2),2*ones(1,imw-2),-ones(1,imw-2),-ones(1,imw-2),2*ones(1,imw-2),-ones(1,imw-2)];

    % inside elements
    for c = 2:imw-1
        idx1 = V(2,c); idx2 = V(imh-1,c);
        i = [i, idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2];
        j = [j, (idx1-imh):(idx2-imh),(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx1+imh):(idx2+imh)];
        v = [v, -ones(1,imh-2),-ones(1,imh-2),4*ones(1,imh-2),-ones(1,imh-2),-ones(1,imh-2)];
    end

    % b matrix
    for c = 1:imw
        for r = 1:imh
            e = e+1;
            if((r==1&&c==1)||(r==1&&c==imw)||(r==imh&&c==1)||(r==imh&&c==imw))
                continue;
            elseif(r==1 || r==imh)
                b(e) = 2*im_s(r,c)-im_s(r,c-1)-im_s(r,c+1);
            elseif(c==1 || c==imw)
                b(e) = 2*im_s(r,c)-im_s(r-1,c)-im_s(r+1,c);
            else    
                b(e) = 4*im_s(r,c)-im_s(r,c-1)-im_s(r,c+1)-im_s(r-1,c)-im_s(r+1,c);
            end
        end
    end

    A = sparse(i,j,v);
    
    %TODO: add extra constraints (if any)

    %TODO: solve the equation
    %use "lscov" or "\", please google the matlab documents
%     solutions(ch) = A \ b;
%     errors(ch) = sum(abs(A*solutions(ch)-b));
end

% disp(errors)

%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
imgout = im_t;
