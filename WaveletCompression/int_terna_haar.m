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
level = 2; % iteration times

% decompostion
img_decomp = I;

for i = 1:level
    img_decomp = haar_2D_decomp(img_decomp);
end
figure;
imshow(img_decomp/255);
imwrite(img_decomp/255,['results/',filename,'_int_terna_haar_decomp_',num2str(level),'.png']);

% reconstruction
img_recon = img_decomp;
for i = 1:level
    img_recon = haar_2D_recon(img_recon);
end
figure;
imshow(img_recon/255);
imwrite(img_recon/255,['results/',filename,'_int_terna_haar_recon_',num2str(level),'.png']);

error = sum(sum(abs(I-img_recon)));
disp(error);

% decomposition 1D function
function img_1D_decomp = haar_1D_decomp(img)  
    n = length(img);
    C = zeros(1,n/3);
    D1 = zeros(1,n/3);
    D2 = zeros(1,n/3);
    for i=1:n/3
        C(i) = floor((img(3*i-2)+img(3*i-1)+img(3*i))/3);
        D1(i) = img(3*i-2)-2*img(3*i-1)+img(3*i);
        D2(i) = img(3*i-2)+img(3*i-1)-2*img(3*i);
    end
    img_1D_decomp = [C,D1,D2];
end

% decomposition 2D function
function img_2D_decomp = haar_2D_decomp(img)
    [rows,cols] = size(img);
    img_2D_decomp = zeros(rows,cols);
    for row = 1:rows       
        img_2D_decomp(row,:) = haar_1D_decomp(img(row,:));
    end
    for col = 1:cols      
        img_2D_decomp(:,col) = (haar_1D_decomp(img_2D_decomp(:,col)))';
    end
end

% reconstruction 1D function
function img_1D_recon = haar_1D_recon(img_1D_decomp) 
    [rows,cols] = size(img_1D_decomp);
    img_1D_recon = zeros(rows,cols);
    C = img_1D_decomp(1, 1:(cols/3));
    D1 = img_1D_decomp(1, (cols/3+1):(2*cols/3));
    D2 = img_1D_decomp(1, (2*cols/3+1):cols);
    for i=1:cols/3
        img_1D_recon(3*i) = C(i)-floor(D2(i)/3);
        img_1D_recon(3*i-1) = C(i)-floor(D1(i)/3);
        img_1D_recon(3*i-2) = D1(i)+2*img_1D_recon(3*i-1)-img_1D_recon(3*i);
    end
end

% reconstruction 2D function
function img_2D_recon = haar_2D_recon(img_2D_decomp)
    [rows, cols] = size(img_2D_decomp);
    img_2D_recon = zeros(rows, cols);
    for col = 1:cols/3      
        img_2D_recon(:,col) = haar_1D_recon(img_2D_decomp(:,col)');  
        img_2D_recon(:,cols/3+col) = haar_1D_recon(img_2D_decomp(:,cols/3+col)') ; 
        img_2D_recon(:,2*cols/3+col) = haar_1D_recon(img_2D_decomp(:,2*cols/3+col)') ; 
    end
    for row = 1:rows/3      
        img_2D_recon(row,:) = haar_1D_recon(img_2D_recon(row,:));
        img_2D_recon(rows/3+row,:) = haar_1D_recon(img_2D_recon(rows/3+row,:)); 
        img_2D_recon(2*rows/3+row,:) = haar_1D_recon(img_2D_recon(2*rows/3+row,:)); 
    end
end
