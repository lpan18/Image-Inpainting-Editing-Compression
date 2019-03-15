clc;
clear;
close all;
imgin = im2double(imread('./target1.jpg'));
[imh, imw, nb] = size(imgin);
assert(nb==1);
% the image is grayscale

V = zeros(imh, imw);
V(1:imh*imw) = 1:imh*imw;
% V(y,x) = (y-1)*imw + x;
% use V(y,x) to represent the variable index of pixel (x,y)
% Always keep in mind that in matlab indexing starts with 1, not 0

%TODO: initialize counter, A (sparse matrix) and b.
e = 0;
b = zeros(imh*imw+4, 1);
num_v = (imh-2)*2*3; % vertical edges 
num_h = (imw-2)*2*3; % horizontal edges
num_in = (imh-2)*(imw-2)*5; % interior elements
num_of_i = num_v+num_h+num_in+4; % 4 contraints
i = zeros(1,num_of_i);
j = zeros(1,num_of_i);
v = zeros(1,num_of_i);

%TODO: fill the elements in A and b, for each pixel in the image
% vertical edges - V(2,1),V(3,1)...V(49,1), V(2,50),V(3,50)...V(49,50)
idx1 = V(2,1); idx2 = V(imh-1,1);
idx3 = V(2,imw); idx4 = V(imh-1,imw);
i(1:num_v) = [idx1:idx2,idx1:idx2,idx1:idx2,idx3:idx4,idx3:idx4,idx3:idx4];
j(1:num_v) = [(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx3-1):(idx4-1),idx3:idx4,(idx3+1):(idx4+1)];
v(1:num_v) = [-ones(1,imh-2),2*ones(1,imh-2),-ones(1,imh-2),-ones(1,imh-2),2*ones(1,imh-2),-ones(1,imh-2)];

% horizonal edges - V(1,2),V(1,3)...V(1,49),V(50,2),V(50,3)...V(50,49)
idx1 = V(1,2); idx2 = V(1,imw-1);
idx3 = V(imh,2); idx4 = V(imh,imw-1);
i((num_v+1):(num_v+num_h)) = [idx1:imh:idx2,idx1:imh:idx2,idx1:imh:idx2,idx3:imh:idx4,idx3:imh:idx4,idx3:imh:idx4];
j((num_v+1):(num_v+num_h)) = [(idx1-imh):imh:(idx2-imh),idx1:imh:idx2,(idx1+imh):imh:(idx2+imh),(idx3-imh):imh:(idx4-imh),idx3:imh:idx4,(idx3+imh):imh:(idx4+imh)];
v((num_v+1):(num_v+num_h)) = [-ones(1,imw-2),2*ones(1,imw-2),-ones(1,imw-2),-ones(1,imw-2),2*ones(1,imw-2),-ones(1,imw-2)];

% interior elements
start_idx = num_v+num_h+1;
for c = 2:imw-1
    idx1 = V(2,c); idx2 = V(imh-1,c);
    delta=(idx2-idx1+1)*5;
    i(start_idx:(start_idx+delta-1)) = [idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2];
    j(start_idx:(start_idx+delta-1)) = [(idx1-imh):(idx2-imh),(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx1+imh):(idx2+imh)];
    v(start_idx:(start_idx+delta-1)) = [-ones(1,imh-2),-ones(1,imh-2),4*ones(1,imh-2),-ones(1,imh-2),-ones(1,imh-2)];
    start_idx = start_idx+delta;
end

% b matrix
for c = 1:imw
    for r = 1:imh
        e = e+1;
        if((r==1&&c==1)||(r==1&&c==imw)||(r==imh&&c==1)||(r==imh&&c==imw))
            continue;
        elseif(r==1 || r==imh)
            b(e) = 2*imgin(r,c)-imgin(r,c-1)-imgin(r,c+1);
        elseif(c==1 || c==imw)
            b(e) = 2*imgin(r,c)-imgin(r-1,c)-imgin(r+1,c);
        else    
            b(e) = 4*imgin(r,c)-imgin(r,c-1)-imgin(r,c+1)-imgin(r-1,c)-imgin(r+1,c);
        end
    end
end

%TODO: add extra constraints
i((num_of_i-3):num_of_i) = [e+1,e+2,e+3,e+4];
j((num_of_i-3):num_of_i) = [1,V(1,imw),V(imh,1),V(imh,imw)];
v((num_of_i-3):num_of_i) = [1,1,1,1]; 

A = sparse(i,j,v);
% similar to the ground truth
b(e+1,1) = imgin(1,1);
b(e+2,1) = imgin(1,imw);
b(e+3,1) = imgin(imh,1);
b(e+4,1) = imgin(imh,imw);

% globally brighter
% b(e+1,1) = imgin(1,1) + 0.3;
% b(e+2,1) = imgin(1,imw) + 0.3;
% b(e+3,1) = imgin(imh,1) + 0.3;
% b(e+4,1) = imgin(imh,imw) + 0.3;

% brighter on the left side
% b(e+1,1) = imgin(1,1) + 0.3;
% b(e+2,1) = imgin(1,imw);
% b(e+3,1) = imgin(imh,1) + 0.3;
% b(e+4,1) = imgin(imh,imw);

% brighter on the bottom side
% b(e+1,1) = imgin(1,1);
% b(e+2,1) = imgin(1,imw);
% b(e+3,1) = imgin(imh,1) + 0.3;
% b(e+4,1) = imgin(imh,imw) + 0.3;

% brighter on right bottom corner
% b(e+1,1) = imgin(1,1);
% b(e+2,1) = imgin(1,imw);
% b(e+3,1) = imgin(imh,1);
% b(e+4,1) = imgin(imh,imw) + 0.3;

%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
solution = A\b;
error = sum(abs(A*solution-b));
disp(error);
imgout = reshape(solution,[imh,imw]);
imwrite(imgout,'output.png');
figure(), hold off, imshow(imgout);