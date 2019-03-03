clc;
clear;
close all;
filename = 'test';
img = im2double(imread([filename,'.jpg']));
[rows,cols,channels] = size(img);
svd_img = zeros(rows,cols,channels);
for ch = 1:channels
    [U, S, V] = svd(img(:,:,ch));
    % drop 5, 10, and 15 smallest singular values
    drop_num = 15;
    s = size(U,2)-drop_num;
    U = U(:,1:s);
    S = S(1:s,1:s);
    V = V';
    V = V(1:s,:);
    svd_img(:,:,ch) = U*S*V;
end
% save compressed image
imwrite(svd_img,[filename,num2str(drop_num),'.jpg']);
%Compute PSNR 
psnr = psnr(svd_img,img) % 122.1126, 112.2993, 106.6234