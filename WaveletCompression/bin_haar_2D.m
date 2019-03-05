clear all;
close all;
clc;
filename = 'Lena';
I = imread([filename,'.png']);
imwrite(I,'test.png');
I = rgb2gray(I);

% create binary image
img = imbinarize(I);

[rows,cols] = size(img);
img_out = zeros(rows, cols);
n = log2(rows);

% initiate wavelet times k (k<=n)
k = 1;
for i = 1:k
    for row = 1:rows       
        [row_C,row_D] = bin_haar_1D(img(row,:));
        img_out(row,:) = [row_C,row_D];
    end
    for col = 1:cols      
        [col_C,col_D] = bin_haar_1D(img_out(:,col)');
        img_out(:,col) = [col_C,col_D]';
    end
    imwrite(img_out,[filename,'_bin_haar_2D_',num2str(i),'.png']);
    CC = img_out(1:rows/2,1:cols/2); 
    DC = img_out(1:rows/2,(cols/2+1):cols);
    CD = img_out((rows/2+1):rows,1:cols/2);
    DD = img_out((rows/2+1):rows,(cols/2+1):cols);
end

function [C,D] = bin_haar_1D(F)  
    n = length(F);
    i = [1:n/2,1:n/2];
    j = [1:2:(n-1),2:2:n];
    v1 = 0.5*ones(1,n);
    v2 = [0.5*ones(1,n/2), -0.5*ones(1,n/2)];
    A1 = sparse(i,j,v1);
    A2 = sparse(i,j,v2);
    C = F*A1';
    D = F*A2';
end