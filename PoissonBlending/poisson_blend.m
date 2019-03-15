function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image
[imh, imw, nb] = size(im_s); % imh = 432, w = 768, nb = 3
% find all non-zero values
[rows,cols] = find(mask_s);
k = size(rows,1); % num of "1" values

V = zeros(imh, imw);
V(1:imh*imw) = 1:imh*imw;

M = containers.Map('KeyType','double','ValueType','double');
for i = 1:k
    M(V(rows(i),cols(i)))=i;
end

i = zeros(1,k*5);
j = zeros(1,k*5);
v = zeros(1,k*5);
b = zeros(k,1);
solutions = zeros(k,nb);

%TODO: consider different channel numbers
for ch = 1:nb
    e = 0;
    for kk = 1:k
        self = 0;
        r = rows(kk);
        c = cols(kk);
        % left
        if(c-1>=1)
            self = self+1;
            if(mask_s(r,c-1) == 1)
                e = e+1;
                i(e) = kk;
                j(e) = M(V(r,c-1));
                v(e) = -1;
                b(kk) = -im_s(r,c-1,ch);
            else
                b(kk) = -im_s(r,c-1,ch)+im_t(r,c-1,ch);
            end
        end
        % right
        if(c+1<=imw)
            self = self+1;
            if(mask_s(r,c+1) == 1)
                e = e+1;
                i(e) = kk;
                j(e) = M(V(r,c+1));
                v(e) = -1;
                b(kk) = b(kk)-im_s(r,c+1,ch);
            else
                b(kk) = b(kk)-im_s(r,c+1,ch)+im_t(r,c+1,ch);
            end
        end
        % up
        if(r-1>=1)
            self = self+1;
            if(mask_s(r-1,c) == 1)
                e = e+1;
                i(e) = kk;
                j(e) = M(V(r-1,c));
                v(e) = -1;
                b(kk) = b(kk)-im_s(r-1,c,ch);
            else
                b(kk) = b(kk)-im_s(r-1,c,ch)+im_t(r-1,c,ch);
            end
        end
        % down
        if(r+1<=imh)
            self = self+1;
            if(mask_s(r+1,c) == 1)
                e = e+1;
                i(e) = kk;
                j(e) = M(V(r+1,c));
                v(e) = -1;
                b(kk) = b(kk)-im_s(r+1,c,ch);
            else
                b(kk) = b(kk)-im_s(r+1,c,ch)+im_t(r+1,c,ch);
            end
        end
        % self
        if(c-1>=1 || c+1<=imw || r-1>=1 || r+1<=imh)
            e = e+1;
            i(e) = kk;
            j(e) = M(V(r,c));
            v(e) = self;
            b(kk) = b(kk) + im_s(r,c,ch)*self;
        end
    end
    i = i(1:e);
    j = j(1:e);
    v = v(1:e);
    
    A = sparse(i,j,v);
    
    %TODO: add extra constraints (if any)

    %TODO: solve the equation
    %use "lscov" or "\", please google the matlab documents
    solution = A\b;
    error = sum(abs(A*solution-b));
    disp(error);
    solutions(:,ch)=solution;
end

%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
imgout = im_t;
for i = 1:k
    imgout(rows(i),cols(i),:) = solutions(i,:);
end