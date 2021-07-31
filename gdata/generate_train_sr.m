clear;
close all;
folder = '../DIV2K_train_HR/';

savepath = '../tr_rgb64/';

%% scale factors
scale = 4;

size_label = 96;
size_input = size_label/scale;
stride = 48;

%% downsizing
downsizes = [1];

data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);

count = 0;
savenum = 0;
margain = 0;

%% generate data
filepaths = [];
filepaths = [filepaths; dir(fullfile(folder, '*.jpg'))];
filepaths = [filepaths; dir(fullfile(folder, '*.bmp'))];
filepaths = [filepaths; dir(fullfile(folder, '*.png'))];

length(filepaths)

for i = 1 : length(filepaths)
    filepaths(i).name
    for flip = 1: 1
        for degree = 1 : 1
            for downsize = 1 : length(downsizes)
                image = imread(fullfile(folder,filepaths(i).name));
                if flip == 1
                    image = flipdim(image ,1);
                end
                if flip == 2
                    image = flipdim(image ,2);
                end
                
                %image = imrotate(image, 90 * (degree - 1));
                image = imresize(image,downsizes(downsize),'bicubic');

                if size(image,3)==3
                    %image = rgb2ycbcr(image);
                    image = im2double(image);
                    im_label = modcrop(image, scale);
                    [hei,wid, c] = size(im_label);
                    
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            subim_label = im_label(x : x+size_label-1, y : y+size_label-1, :);
                            subim_input = imresize(subim_label,1/scale,'bicubic');
                            % figure;
                            % imshow(subim_input);
                            % figure;
                            % imshow(subim_label);
                            count=count+1;
                            data(:, :, :, count) = subim_input;
                            label(:, :, :, count) = subim_label;
                        end
                    end
                end
            end
        end
    end
    if mod(i,16)==0
        savefile = [savepath,'d64_train_',num2str(savenum,'%08d'),'.h5']
        order = randperm(count);
        data = data(:, :, :, order);
        label = label(:, :, :, order); 

        %% writing to HDF5
        chunksz = 64;
        created_flag = false;
        totalct = 0;

        for batchno = 1:floor(count/chunksz)
            batchno;
            last_read=(batchno-1)*chunksz;
            batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
            batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
            startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
            curr_dat_sz = store2hdf5(savefile, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
            created_flag = true;
            totalct = curr_dat_sz(end);
        end

        h5disp(savefile);
        savenum = savenum + 1;
        count = 0;
    end
end