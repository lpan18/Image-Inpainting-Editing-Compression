function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image

[imh, imw, nb] = size(im_s); % imh = 567, imw = 988, nb = 3

%TODO: consider different channel numbers

%TODO: initialize counter, A (sparse matrix) and b.
%Note: A don't have to be k¡Ák,
%      you can add useless variables for convenience,
%      e.g., a total of imh*imw variables
k = imh*imw
A = sparse(k+4,k)
b = zeros(k+4,1);

%TODO: fill the elements in A and b, for each pixel in the image
i = zeros(1,2*k+4);
j = zeros(1,2*k+4);
v = zeros(1,2*k+4);
i(1:k) = 1:k;
j(1:k) = 1:k;
v(1:k) = -1;
i(k+1:2*k) = 1:k;
j(k+1:2*k) = 2:k+1;
v(k+1:2*k) = 1;

%TODO: add extra constraints (if any)
i(2*k+1) = 1;
j(2*k+1) = 1;
v(2*k+1) = 1;

i(2*k+2) = 1;
j(2*k+2) = k;
v(2*k+2) = 1;

i(2*k+2) = k;
j(2*k+2) = 1;
v(2*k+2) = 1;

i(2*k+2) = k;
j(2*k+2) = k;
v(2*k+2) = 1;

A = sparse(i, j, v)

%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
solution = A \ b;
error = sum(abs(A*solution-b));
disp(error)

%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
imgout = im_t;
