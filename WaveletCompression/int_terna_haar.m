% Integer Ternary Haar Wavelet Decomposition and Reconstruction
clear all;
close all;
clc;

filename = 'Lena';
I = imread([filename,'.jpg']);
if(ndims(I)==3)
    I = rgb2gray(I);
end
I = double(I); % in matlab 
I = imcrop(I,[134 134 242 242]); % crop image to be 243*243
[rows,cols] = size(I);
n = log2(rows);
level = 3; % iteration times

% decompostion
img_decomp = I;
for i = 1:level
    [CC,DC,CD,DD] = haar_2D_decomp(img_decomp);
    img_decomp=[CC,DC;CD,DD];
end
figure;
imshow(img_decomp/255);
imwrite(img_decomp/255,['results/',filename,'_int_terna_haar_decomp_',num2str(level),'.png']);

% reconstruction
img_recon = img_decomp;
[rows,cols] = size(img_recon);
for i = 1:level
    CC = img_recon(1:rows/3,1:cols/3); 
    DC = img_recon(1:rows/3,(cols/3+1):cols);
    CD = img_recon((rows/3+1):rows,1:cols/3);
    DD = img_recon((rows/3+1):rows,(cols/3+1):cols);
    img_recon = haar_2D_recon(CC,DC,CD,DD);
end
figure;
imshow(img_recon/255);
imwrite(img_recon/255,['results/',filename,'_int_terna_haar_recon_',num2str(level),'.png']);

% decomposition 1D function
function [C,D] = haar_1D_decomp(img)  
    n = length(img);
    C = zeros(1,n/3);
    D = zeros(1,2*n/3);
    for i=1:n/3
        C(i) = floor((img(3*i-2)+img(3*i-1)+img(3*i))/3);
        D(2*i-1) = img(3*i-2)-2*img(3*i-1)+img(3*i);
        D(2*i) = img(3*i-2)+img(3*i-1)-2*img(3*i);
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
    CC = img(1:rows/3,1:cols/3); 
    DC = img(1:rows/3,(cols/3+1):cols);
    CD = img((rows/3+1):rows,1:cols/3);
    DD = img((rows/3+1):rows,(cols/3+1):cols);
end

% reconstruction 1D function
function img_1D_recon = haar_1D_recon(C,D)  
    n = length(C);
    img_1D_recon = zeros(1,3*n);
    for i=1:n
        img_1D_recon(3*i) = C(i)-floor(D(2*i)/3);
        img_1D_recon(3*i-1) = C(i)-floor(D(2*i-1)/3);
        img_1D_recon(3*i-2) = D(2*i-1)+2*img_1D_recon(3*i-1)-img_1D_recon(3*i);
    end
end

% reconstruction 2D function
function img_2D_recon = haar_2D_recon(CC,DC,CD,DD)
    [rows, cols] = size(CC);
    img_2D_recon = zeros(3*rows, 3*cols);
    for col = 1:cols      
        img_2D_recon(:,col) = haar_1D_recon(CC(:,col),CD(:,col));  
        img_2D_recon(:,cols+col) = haar_1D_recon(DC(:,col),DD(:,col)) ; 
        img_2D_recon(:,2*cols+col) = haar_1D_recon(DC(:,cols+col),DD(:,cols+col)) ; 
    end
    for row = 1:rows      
        img_2D_recon(row,:) = haar_1D_recon(img_2D_recon(row,1:cols),img_2D_recon(row,(cols+1):3*cols));
        img_2D_recon(rows+row,:) = haar_1D_recon(img_2D_recon(rows+row,1:cols),img_2D_recon(rows+row,(cols+1):3*cols)); 
        img_2D_recon(2*rows+row,:) = haar_1D_recon(img_2D_recon(2*rows+row,1:cols),img_2D_recon(2*rows+row,(cols+1):3*cols)); 
    end
end
