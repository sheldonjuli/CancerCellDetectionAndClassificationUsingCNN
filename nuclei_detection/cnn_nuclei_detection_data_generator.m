
% This file generates data necessary to train detection NN
% Put Detection of crchistophenotypes_2016_04_28 in the same directory
% Data will be generated in TRAIN_DIR and VAL_DIR
% In VAL_DIRTRAIN_DIR and VAL_DIR there will be 2 folder (1 and 2)

clear variables;
close all;
clc;

% Set random seed
rng('default') ;
rng(0) ;

% The sets, and number of samples per label in each set
sets = {'train', 'val'};
labels = {'1', '2'}; % 1 mean not a nucleus

TRAIN_DIR = '../../data/nuclei-dataset/train';
VAL_DIR = '../../data/nuclei-dataset/val';

% Delete old folders before running
mkdir(sprintf('%s/%s', TRAIN_DIR, '1'));
mkdir(sprintf('%s/%s', TRAIN_DIR, '2'));
mkdir(sprintf('%s/%s', VAL_DIR, '1'));
mkdir(sprintf('%s/%s', VAL_DIR, '2'));

count = 0;
train_count = 0;
val_count = 0;

tic;
for j = 1 : 100
       
    fprintf("Collected %d. Collecting samples from img%d.\n", count, j);
    data = load_detections(j);
    img = data.img;
    detec = data.detection.detection;

    each_count = 0;
    
    % Collect y data
    for i = 1 : size(detec)
        
        % Collect 1500 * 2 for training (1500 for '1' and 1500 for '2')
        % Collect 150 * 2 for evaluation (150 for '1' and 150 for '2')
        % Collect no more than 100 from a single image
        % Collect from images with detection less than 175 (to avoid long running time for collecting '1')
        if (train_count >= 3200 && val_count >= 320) || each_count >= 250 || size(detec,1) > 225
            break
        end
        
        c_p = detec(i, 1:2);
        c_x = c_p(1, 1);
        c_y = c_p(1, 2);
        
        % Avoid edges
        if c_x >= 14 && c_x <= 486 && c_y >= 14 && c_y <= 486
            each_count = each_count + 1;
            point = getRandom();
            while closeToDetection(point, detec)
                point = getRandom();
            end
        
            count = count + 1;
            wind = imcrop(img, [int16(c_x - 13.5), int16(c_y - 13.5), 26, 26]);
            wind_n = imcrop(img, [int16(point.x - 13.5), int16(point.y - 13.5), 26, 26]);
            %figure, imshow(wind)

            if mod(count, 10) == 0 && val_count <= 299
                val_count = val_count + 1;
                img_name = sprintf('%s/2/%d.jpg', VAL_DIR, val_count);
                img_name_n = sprintf('%s/1/%d.jpg', VAL_DIR, val_count);
            else
                train_count = train_count + 1;
                img_name = sprintf('%s/2/%d.jpg', TRAIN_DIR, train_count);
                img_name_n = sprintf('%s/1/%d.jpg', TRAIN_DIR, train_count);
            end
            imwrite(rgb2gray(wind), img_name, 'jpg');
            imwrite(rgb2gray(wind_n), img_name_n, 'jpg');
        end
                
    end

end
e = toc;

fprintf('Collected %d in total. %d train data and %d test data (took: %0.4f seconds)\n', count, train_count, val_count, e);

% Check if the given point is close to an existing detection
function result = closeToDetection(point, detec)

    acceptable_dist = 3.5;

    result = 0;
    for i = 1 : size(detec)
        
        c_p = detec(i, 1:2);
        c_x = c_p(1, 1);
        c_y = c_p(1, 2);
        
        % 
        if abs(point.x - c_x) <= acceptable_dist || abs(point.y - c_y) <= acceptable_dist
            result = 1;
            break
        end
    end
end

% Get a random xy coordinate
function point = getRandom()
    % avoid edges
    R = [14 486];
    z = rand(2,1)*range(R)+min(R);
    point.x = z(1);
    point.y = z(2);
end

% Load data
function data = load_detections(img_id)
    RAW_IMG_DIR = './Detection';
    img_dir = sprintf('%s/img%d', RAW_IMG_DIR, img_id);
    files = dir(fullfile(img_dir, sprintf('/img%d*', img_id)));
    if isempty(files)
        fprintf('file doesn''t exist!\n');
    else
        data.img = imread(fullfile(img_dir, files(1).name));
        data.detection = load(fullfile(img_dir, files(2).name));
    end
end
