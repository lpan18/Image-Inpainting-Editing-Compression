% Binary Haar Wavelet Decomposition and Reconstruction
clear all;
close all;
clc;

filename = 'Lena';
I = imread([filename,'.jpg']);
if(ndims(I)==3)
    I = rgb2gray(I);
end
I = im2double(I);
[rows,cols] = size(I);
n = log2(rows);
level = 1; % iteration times

% decompostion
img_decomp = I;
for i = 1:level
    [CC,DC,CD,DD] = haar_2D_decomp(img_decomp);
    img_decomp=[CC,DC;CD,DD];
end
figure;
imshow(img_decomp+0.5);
imwrite(img_decomp+0.5,['results/',filename,'_bin_haar_decomp_',num2str(level),'.png']);

% reconstruction
img_recon = img_decomp;
[rows,cols] = size(img_recon);
for i = 1:level
    CC = img_recon(1:rows/2,1:cols/2); 
    DC = img_recon(1:rows/2,(cols/2+1):cols);
    CD = img_recon((rows/2+1):rows,1:cols/2);
    DD = img_recon((rows/2+1):rows,(cols/2+1):cols);
    img_recon = haar_2D_recon(CC,DC,CD,DD);
end
figure;
imshow(img_recon);
imwrite(img_recon,['results/',filename,'_bin_haar_recon_',num2str(level),'.png']);

% decomposition 1D function
function [C,D] = haar_1D_decomp(img)  
    n = length(img);
    C = zeros(1,n/2);
    D = zeros(1,n/2);
    for i=1:n/2
        C(i) = (img(2*i-1)+img(2*i))/2;
        D(i) = (img(2*i-1)-img(2*i))/2;
    end
end

% decomposition 2D function
function [CC,DC,CD,DD] = haar_2D_decomp(img)
    [rows,cols] = size(img);
    for row = 1:rows       
        [row_C,row_D] = haar_1D_decomp(img(row,:));
        img(row,:) = [row_C,row_D];
    end
    for col = 1:cols      
        [col_C,col_D] = haar_1D_decomp(img(:,col));
        img(:,col) = [col_C,col_D]';
    end
    CC = img(1:rows/2,1:cols/2); 
    DC = img(1:rows/2,(cols/2+1):cols);
    CD = img((rows/2+1):rows,1:cols/2);
    DD = img((rows/2+1):rows,(cols/2+1):cols);
end

% reconstruction 1D function
function img_1D_recon = haar_1D_recon(C,D)  
    n = length(C);
    img_1D_recon = zeros(1,2*n);
    for i=1:n
        img_1D_recon(2*i-1) = C(i)+D(i);
        img_1D_recon(2*i) = C(i)-D(i);
    end
end

% reconstruction 2D function
function img_2D_recon = haar_2D_recon(CC,DC,CD,DD)
    [rows, cols] = size(CC);
    img_2D_recon = zeros(2*rows, 2*cols);
    for col = 1:cols      
        img_2D_recon(:,col) = haar_1D_recon(CC(:,col),CD(:,col));  
        img_2D_recon(:,cols+col) = haar_1D_recon(DC(:,col),DD(:,col)) ; 
    end
    for row = 1:rows      
        img_2D_recon(row,:) = haar_1D_recon(img_2D_recon(row,1:cols),img_2D_recon(row,(cols+1):2*cols));
        img_2D_recon(rows+row,:) = haar_1D_recon(img_2D_recon(rows+row,1:cols),img_2D_recon(rows+row,(cols+1):2*cols)); 
    end
end