clc;
clear;
close all;

% image compression
img = im2double(imread('./test.jpg'));
% imshow(img)
img = rgb2gray(img);
img = imresize(img,[1000 1000]);
s= 10;
[U, S, V] = svd(img);
U = U(:,1:s);
S = S(1:s,1:s);
V = V';
V = V(1:s,:);
img = U*S*V;
figure();
imshow(img);



load('faces.mat');
% images: size=[200,300,250], containing 200 images [250x300]

images = images/255;
face_1 = squeeze(images(1,:,:));

% 
% mean = mean(face, 1);
% face = face - mean
%--------TODO--------
%PCA the faces and show the eigenfaces.
% [U, S, V] = svd(face, 'econ');

%Select the first 100 eigenfaces
% U = U(:, 1:100);
%Load an image and project it to the "face space" to get a vector representing this image
galaxy = galaxy * V;

%Reconstruct the image by the vector and eigenfaces

%Compute PSNR 