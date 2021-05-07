% removed course code and renamed images to prevent plagarism
% Marcin Lenart
clear all; close all; clc;

%read raw image data to fix
raw_img=imread('input/test.png');


%original good image for error comparison
original_img=imread('rgb/test.png');

%4 images for machine learning
%use resize function if sizes are different
%learning_img_raw_1 = imresize(imread('test2.jpg'), [256,384]);
learning_img_raw_1=imread('rgb/1.png');
learning_img_raw_2=imread('rgb/2.png');
learning_img_raw_3=imread('rgb/3.png');
learning_img_raw_4=imread('rgb/4.png');

%concatenate learning images
learning_img = cat(2, learning_img_raw_1, learning_img_raw_2,learning_img_raw_3,learning_img_raw_4);

%uncomment this for single learning image
%learning_img=imread('rgb/test.png');

%display input image
figure;
imshow(raw_img);
title('Raw Input Image');

%determine both of the image dimensions
[num_of_rows,num_of_cols,layers]=size(raw_img);
[num_of_rows2,num_of_cols2,layers2]=size(learning_img);

%pass images through the rgb bayer filter function
bayer_img=getBayerImg(raw_img);
bayer_learning_img=getBayerImg(learning_img);

%output of the bayer image before interpolation
figure
imshow(bayer_img)
title('Bayer Image');

%output original learning image
figure
imshow(learning_img)
title('Learning Image');

%format image for matrix use ** put this in function later
learning_img_dbl=double(learning_img); %convert learning img to double
bayer_learning_img=double(bayer_learning_img); %convert bayer learning img to double
learning_img_1layer=sum(bayer_learning_img,3); %convert learning img to single layer image

%expand borders function to prevent edge problems
learning_img_1layer_exp=expandBorders(learning_img_1layer);

%output of expanded, 1 layer learning image
figure
imshow(uint8(learning_img_1layer_exp))
title('Learning Image -1 Layer, Expanded Border');

%Machine Learning Section__________________________________________________
%preallocate 4 X-matrices
num_of_elements=(num_of_rows2*num_of_cols2/4);
x_matrix=zeros(num_of_elements,49,4);

%index incrementors
x1=0;
x2=0;
x3=0;
x4=0;

%preallocate y-matrix (true values)
y=zeros(num_of_elements,8);

%calculate X-matrix
%scan each pixel of learning image(offset by 3 pixels to get center pixel of expanded image)
for row =1+3:num_of_rows2+3
    for col =1+3:num_of_cols2+3
        Sub_x_matrix = learning_img_1layer_exp(row-3:row+3,col-3:col+3); %extract sub matrix (7x7)
        Sub_x_matrix=reshape(Sub_x_matrix.',1,[]); %convert to a row vector
        if (mod(row,2)==0 && mod(col,2)==0)
            %red tile (even row, even column)
            x1=x1+1;
            x_matrix( x1, :, 1 ) = Sub_x_matrix; % update to x-matrix 1
            %update y-pixel true value matrix
            y(x1,1)=learning_img_dbl(row-3,col-3,2);%green1
            y(x1,2)=learning_img_dbl(row-3,col-3,3);%blue1
        elseif mod(row,2)==0 && mod(col,2)==1
            %green tile-first row (even row, odd column
            x2=x2+1;
            x_matrix( x2, :, 2 ) = Sub_x_matrix; % update to x-matrix 2
            y(x2,3)=learning_img_dbl(row-3,col-3,1);%red1
            y(x2,4)=learning_img_dbl(row-3,col-3,3);%blue2
        elseif mod(row,2)==1 && mod(col,2)==0
            %green tile-second row (odd row, even column
            x3=x3+1;
            x_matrix( x3, :, 3 ) = Sub_x_matrix; % update to x-matrix 3
            y(x3,5)=learning_img_dbl(row-3,col-3,1);%red2
            y(x3,6)=learning_img_dbl(row-3,col-3,3);%blue3
        else
            %blue tile (odd row, odd column)
            x4=x4+1;
            x_matrix( x4, :, 4 ) = Sub_x_matrix; % update to x-matrix 4
            y(x4,7)=learning_img_dbl(row-3,col-3,1);%red3
            y(x4,8)=learning_img_dbl(row-3,col-3,2);%green2
        end
    end
end

%compute A coefficients (8 sets)
A=zeros(49,8); %preallocate A-matrix into 8 column vectors
for index=1:8
    %select required X-matrix
    x_index=round(index/2);%adjusts index range to 1-4
    X=x_matrix(:,:,x_index);
    x_trans=transpose(X);
    
    %calculation for A coefficients
    A(:,index)=inv(x_trans*X)*x_trans*y(:,index);
end

disp('Machine Learning Done...Demosaicing Started...');
%Demosaicing Section_______________________________________________________
%format image to be demosaiced
bayer_img_dbl=double(bayer_img); %convert bayer image int8 to double
demosaic_img=bayer_img_dbl; %create a copy for final demosaic image
bayer_img_1layer=sum(bayer_img_dbl,3); %convert bayer to single layer image

%expand borders for calculation section
bayer_img_1layer_exp=expandBorders(bayer_img_1layer);

%  figure;
%  imshow(uint8(bayer_img_1layer_exp));
%  title('expanded 1 layer exp Image');

%Demosaic using X*A formula, store final in demosaic_img
%scan each pixel (offset by 3 pixels to get center pixel of expanded image)
offset=-3;%pixel offset for non-expanded image
for row =1+3:num_of_rows+3
    for col =1+3:num_of_cols+3
        Sub_x_matrix = bayer_img_1layer_exp(row-3:row+3,col-3:col+3); %extract sub matrix (7x7)
        Sub_x_matrix=reshape(Sub_x_matrix.',1,[]); %(convert to a row vector)
        if (mod(row,2)==0 && mod(col,2)==0)
            %red tile (even row, even column)
            %use corresponding A-matrix (1-8) depending on type of tile
            demosaic_img(row+offset,col+offset,2)=Sub_x_matrix*A(:,1);
            demosaic_img(row+offset,col+offset,3)=Sub_x_matrix*A(:,2);
            
        elseif mod(row,2)==0 && mod(col,2)==1
            %green tile-first row (even row, odd column
            demosaic_img(row+offset,col+offset,1)=Sub_x_matrix*A(:,3);
            demosaic_img(row+offset,col+offset,3)=Sub_x_matrix*A(:,4);
        elseif mod(row,2)==1 && mod(col,2)==0
            %green tile-second row (odd row, even column
            demosaic_img(row+offset,col+offset,1)=Sub_x_matrix*A(:,5);
            demosaic_img(row+offset,col+offset,3)=Sub_x_matrix*A(:,6);
        else
            %blue tile (odd row, odd column)
            demosaic_img(row+offset,col+offset,1)=Sub_x_matrix*A(:,7);
            demosaic_img(row+offset,col+offset,2)=Sub_x_matrix*A(:,8);
        end
    end
end

%Output and Error Section__________________________________________________
demosaic_img=uint8(demosaic_img);%convert back to uint8

%display original image
figure;
imshow(original_img);
title('Original Full-resolution Image');

%display final interpolated image
figure;
imshow(demosaic_img);
title('Demosaiced Image');

%matlab mean square error
err = immse(demosaic_img,original_img)

%alternative mean squared error checker
err1 = (double(original_img) - double(demosaic_img)) .^ 2;
err = sum(sum(err1)) / (num_of_rows * num_of_cols) %error per color layer
err = sum(err)/3 %total error

%display error image
figure;
imshow(uint8(err1));
title('Error Image');

%Functions_________________________________________________________________

%converts to a bayer image
function output_img=getBayerImg(input_image)
%initialize bayer filter to the size of the raw image
[num_of_rows_temp,num_of_cols_temp,~]=size(input_image);
bayer_filter_rgb=input_image*0; %preallocate filter

%generate bayer mosaic patterns in rgb format
%the same uint8 format as the raw image
for row =1:num_of_rows_temp
    for col =1:num_of_cols_temp
        %red
        if mod(row,2)==1 && mod(col,2)==1
            bayer_filter_rgb(row,col,1)=255;
            %blue
        elseif mod(row,2)==0 && mod(col,2)==0
            bayer_filter_rgb(row,col,3)=255;
            %green
        else
            bayer_filter_rgb(row,col,2)=255;
        end
    end
end
output_img= input_image.*(bayer_filter_rgb/255);
end

%expand image borders by mirroring neighbouring pixels
%expands by 3 pixels on each side of original image
function expanded_image=expandBorders(input_image)
[num_of_rows_temp,num_of_cols_temp,~]=size(input_image);
%preallocate image matrix to expanded size and copy original with offset
expanded_image=zeros(num_of_rows_temp+6,num_of_cols_temp+6);
expanded_image(4:num_of_rows_temp+3,4:num_of_cols_temp+3)=input_image;

%add mirrored elements
%left and right mirror components
for col =1:3
    expanded_image(4:num_of_rows_temp+3,col:col)=input_image(1:num_of_rows_temp,5-col);
    expanded_image(4:num_of_rows_temp+3,col+num_of_cols_temp+3:col+num_of_cols_temp+3)=input_image(1:num_of_rows_temp,num_of_cols_temp-col);
end

%top and bottom mirror components
for row =1:3
    expanded_image(row:row,1:num_of_cols_temp+6)=expanded_image(8-row:8-row,1:num_of_cols_temp+6);
    expanded_image(row+num_of_rows_temp+3:row+num_of_rows_temp+3,1:num_of_cols_temp+6)=expanded_image(3+num_of_rows_temp-row:3+num_of_rows_temp-row,1:num_of_cols_temp+6);
end
end



