clear;
close all;
% folder = '../oni/depth/train/';
folder = '../depthbin/noise2/';
save_folder = '../depthbin/label/';
savepath = '../tr_depth32/';

%% scale factors

size_label = 64;
stride = 48;

%% downsizing
downsizes = [1];

label = zeros(size_label, size_label, 1, 1);

count = 0;
savenum = 1;
margain = 0;

desiredMin = 0;
desiredMax = 255;
desiredRange = desiredMax - desiredMin;


%% generate data
filepaths1 = [];
filepaths1 = [filepaths1; dir(fullfile(folder, '*.jpg'))];
filepaths1 = [filepaths1; dir(fullfile(folder, '*.bmp'))];
filepaths1 = [filepaths1; dir(fullfile(folder, '*.png'))];

length(filepaths1)

for i = 1 : length(filepaths1)
    filepaths1(i).name
    for flip = 1: 1
        for degree = 1 : 1
            for downsize = 1 : length(downsizes)
                image = imread(fullfile(folder,filepaths1(i).name));
                image=image(:,:,1);
%                 image = crop_image(image);
%                 image=image(51:466,41:600);
%                 image=double(image/255);
%                 image=image/(max(max(image)))*255;
%                 image=uint8(image);
%                 imshow(image);
%                 image=uint8(image/25);
%                 originalMinValue = double(min(min(image)));
%                 originalMaxValue = double(max(max(image)));
%                 originalRange = originalMaxValue - originalMinValue;
%                 image = uint8(desiredRange * (double(image) - originalMinValue) / originalRange + desiredMin);
%                 imshow(image);
%                 imwrite(image,fullfile(save_folder,filepaths1(i).name));
                if flip == 2
                    image = flip(image ,2);
                end
                
                %image = imrotate(image, 90 * (degree - 1));
                image = imresize(image,downsizes(downsize),'bicubic');

                %if size(image,3)==3
                    %image = rgb2ycbcr(image);
                    image = im2double(image);

                    im_label = image;
                    hole = min(image(:));
                    [hei,wid, c] = size(im_label);
                    
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                            rr=min(subim_label(:));
                            count=count+1;
                            label(:, :,  :,count) = subim_label;
%                             if rr~=0
%                                 count=count+1;
%                                 label(:, :,  :,count) = subim_label;
%                             end
                        end
                    end
                %end
            end
        end
    end
    if mod(i,length(filepaths1))==0
        savefile = [savepath,'d32_train_',num2str(savenum,'%08d'),'.h5'];
        a=size(label);
%         order = randperm(a(4));
%         label = label(:, :, :, order); 

        %% writing to HDF5
        chunksz = 64;
        created_flag = false;
        totalct = 0;

        for batchno = 1:floor(a(4)/chunksz)
            batchno;
            last_read=(batchno-1)*chunksz;
            batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
            startloc = struct('lab', [1,1,1,totalct+1]);
            curr_dat_sz = stored2hdf5(savefile, batchlabs, ~created_flag, startloc, chunksz); 
            created_flag = true;
            totalct = curr_dat_sz(end);
        end

        h5disp(savefile);
        savenum = savenum + 1;
        count = 0;
    end
end