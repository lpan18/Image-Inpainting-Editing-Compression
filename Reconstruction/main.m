clc;
clear;
close all;

imgin = im2double(imread('./target.jpg'));
[imh, imw, nb] = size(imgin);
assert(nb==1);
% the image is grayscale

V = zeros(imh, imw);
V(1:imh*imw) = 1:imh*imw;
% V(y,x) = (y-1)*imw + x;
% use V(y,x) to represent the variable index of pixel (x,y)
% Always keep in mind that in matlab indexing starts with 1, not 0

%TODO: initialize counter, A (sparse matrix) and b.
% the counter e represent the row of A matrix
e = 1;
A = zeros(imh*(imw-1)+(imh-1)*imw+1, imh*imw);
b = zeros(imh*(imw-1)+(imh-1)*imw+1, 1);

%TODO: fill the elements in A and b, for each pixel in the image
% x direction, r,c correspond to the row and column of image
for r = 1:imh
    for c = 1:imw-1
        idx = V(r,c);
        A(e,idx) = -1;
        A(e,idx+imw) = 1;
        b(e) = -imgin(r,c)+imgin(r,c+1);
        e = e+1;
    end
end

% y direction
for r = 1:imh-1
    for c = 1:imw
        idx = V(r,c);
        A(e,idx) = -1;
        A(e,idx+1) = 1;        
        b(e) = -imgin(r,c)+imgin(r+1,c);
        e = e+1;
    end
end

%TODO: add extra constraints
A(e,1) = 1;
b(e,1) = imgin(1,1);

%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
solution = A\b;
error = sum(abs(A*solution-b));
disp(error);
imgout = reshape(solution,[imh,imw]);
imwrite(imgout,'output.png');
figure(), hold off, imshow(imgout);

