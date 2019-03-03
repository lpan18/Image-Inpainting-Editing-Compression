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
b = zeros(k+4, 1);

%TODO: fill the elements in A and b, for each pixel in the image
% vertical edges - V(2,1),V(3,1)...V(49,1), V(2,50),V(3,50)...V(49,50)
idx1 = V(2,1); idx2 = V(imh-1,1);
idx3 = V(2,imw); idx4 = V(imh-1,imw);
i = [idx1:idx2,idx1:idx2,idx1:idx2,idx3:idx4,idx3:idx4,idx3:idx4];
j = [(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx3-1):(idx4-1),idx3:idx4,(idx3+1):(idx4+1)];
v = [-ones(1,imh-2),2*ones(1,imh-2),-ones(1,imh-2),-ones(1,imh-2),2*ones(1,imh-2),-ones(1,imh-2)];

% horizonal edges - V(1,2),V(1,3)...V(1,49),V(50,2),V(50,3)...V(50,49)
idx1 = V(1,2); idx2 = V(1,imw-1);
idx3 = V(imh,2); idx4 = V(imh,imw-1);
i = [i, idx1:imh:idx2,idx1:imh:idx2,idx1:imh:idx2,idx3:imh:idx4,idx3:imh:idx4,idx3:imh:idx4];
j = [j, (idx1-imh):imh:(idx2-imh),idx1:imh:idx2,(idx1+imh):imh:(idx2+imh),(idx3-imh):imh:(idx4-imh),idx3:imh:idx4,(idx3+imh):imh:(idx4+imh)];
v = [v, -ones(1,imw-2),2*ones(1,imw-2),-ones(1,imw-2),-ones(1,imw-2),2*ones(1,imw-2),-ones(1,imw-2)];

% inside elements
for c = 2:imw-1
    idx1 = V(2,c); idx2 = V(imh-1,c);
    i = [i, idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2,idx1:idx2];
    j = [j, (idx1-imh):(idx2-imh),(idx1-1):(idx2-1),idx1:idx2,(idx1+1):(idx2+1),(idx1+imh):(idx2+imh)];
    v = [v, -ones(1,imh-2),-ones(1,imh-2),4*ones(1,imh-2),-ones(1,imh-2),-ones(1,imh-2)];
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
i = [i,k+1,k+2,k+3,k+4];
j = [j,1,V(1,imw),V(imh,1),V(imh,imw)];
v = [v,1,1,1,1]; 

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
% b(e+3,1) = imgin(imh,1) + 0.3;
% brighter on the bottom side
% b(e+3,1) = imgin(imh,1) + 0.3;
% b(e+4,1) = imgin(imh,imw) + 0.3;
% brighter on right bottom corner
b(e+4,1) = imgin(imh,imw) + 0.3;

%TODO: solve the equation
%use "lscov" or "\", please google the matlab documents
solution = A\b;
error = sum(abs(A*solution-b));
disp(error);
imgout = reshape(solution,[imh,imw]);
imwrite(imgout,'output_right_bottom.png');
figure(), hold off, imshow(imgout);

% another solution not using sparse
% e = 0;
% k = imh*imw;
% A = zeros(k+4, k);
% b = zeros(k+4, 1);
% for c = 1:imw
%     for r = 1:imh
%         e = e+1;
%         idx = V(r,c); 
%         if((r==1&&c==1)||(r==1&&c==imw)||(r==imh&&c==1)||(r==imh&&c==imw))
%             continue;
%         elseif(r==1 || r==imh)
%             A(e,idx) = 2;
%             A(e,idx-imh) = -1;
%             A(e,idx+imh) = -1;
%             b(e) = 2*imgin(r,c)-imgin(r,c-1)-imgin(r,c+1);
%         elseif(c==1 || c==imw)
%             A(e,idx) = 2;
%             A(e,idx-1) = -1;
%             A(e,idx+1) = -1;
%             b(e) = 2*imgin(r,c)-imgin(r-1,c)-imgin(r+1,c);
%         else    
%             A(e,idx) = 4;
%             A(e,idx-1) = -1;
%             A(e,idx+1) = -1;
%             A(e,idx-imh) = -1;
%             A(e,idx+imh) = -1;            
%             b(e) = 4*imgin(r,c)-imgin(r,c-1)-imgin(r,c+1)-imgin(r-1,c)-imgin(r+1,c);
%         end
%     end
% end
% A(e+1,1) = 1;
% b(e+1,1) = imgin(1,1);
% A(e+2,V(1,imw)) = V(1,imw);
% b(e+2,1) = imgin(1,imw);
% A(e+3,V(imh,1)) = V(imh,1);
% b(e+3,1) = imgin(imh,1);
% A(e+4,V(imh,imw)) = V(imh,imw);
% b(e+4,1) = imgin(imh,imw);
