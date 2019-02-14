function imgout = poisson_blend(im_s, mask_s, im_t)
% -----Input
% im_s     source image (object)
% mask_s   mask for source image (1 meaning inside the selected region)
% im_t     target image (background)
% -----Output
% imgout   the blended image

[imh, imw, nb] = size(im_s);

%TODO: consider different channel numbers

%TODO: initialize counter, A (sparse matrix) and b.
%Note: A don't have to be k¡Ák,
%      you can add useless variables for convenience,
%      e.g., a total of imh*imw variables
%-----
%-----

%TODO: fill the elements in A and b, for each pixel in the image
%-----
%-----

%TODO: add extra constraints (if any)
%-----
%-----


%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
solution = ???
error = sum(abs(A*solution-b));
disp(error)

%TODO: copy those variable pixels to the appropriate positions
%      in the output image to obtain the blended image
%-----
%-----
