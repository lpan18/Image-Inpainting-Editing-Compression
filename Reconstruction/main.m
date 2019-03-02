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
e = 0;
k = imh*imw;
A = zeros(k+4, k);
b = zeros(k+4, 1);

%TODO: fill the elements in A and b, for each pixel in the image
for c = 1:imw
    for r = 1:imh
        e = e+1;
        idx = V(r,c); 
        if((r==1&&c==1)||(r==1&&c==imw)||(r==imh&&c==1)||(r==imh&&c==imw))
            continue;
        elseif(r==1 || r==imh)
            A(e,idx) = 2;
            A(e,idx-imw) = -1;
            A(e,idx+imw) = -1;
            b(e) = 2*imgin(r,c)-imgin(r,c-1)-imgin(r,c+1);
        elseif(c==1 || c==imw)
            A(e,idx) = 2;
            A(e,idx-1) = -1;
            A(e,idx+1) = -1;
            b(e) = 2*imgin(r,c)-imgin(r-1,c)-imgin(r+1,c);
        else    
            A(e,idx) = 4;
            A(e,idx-1) = -1;
            A(e,idx+1) = -1;
            A(e,idx-imw) = -1;
            A(e,idx+imw) = -1;            
            b(e) = 4*imgin(r,c)-imgin(r,c-1)-imgin(r,c+1)-imgin(r-1,c)-imgin(r+1,c);
        end
    end
end

%TODO: add extra constraints
A(e+1,1) = 1;
b(e+1,1) = imgin(1,1);
A(e+2,V(1,imw)) = V(1,imw);
b(e+2,1) = imgin(1,imw);
A(e+3,V(imh,1)) = V(imh,1);
b(e+3,1) = imgin(imh,1);
A(e+4,V(imh,imw)) = V(imh,imw);
b(e+4,1) = imgin(imh,imw);

%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
solution = A\b;
error = sum(abs(A*solution-b));
disp(error);
imgout = reshape(solution,[imh,imw]);
imwrite(imgout,'output.png');
figure(), hold off, imshow(imgout);

