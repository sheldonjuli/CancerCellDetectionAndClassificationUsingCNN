
% Classify nuclei given their detection points and plot the result
% Put Detection of crchistophenotypes_2016_04_28 in the same directory

clear variables;
close all;
clc;

% Load a model and upgrade it to MatConvNet current version.
trained_data = load('./detec-net-epoch-10.mat') ;
net = trained_data.net;
net = vl_simplenn_tidy(net) ;
net.layers{end}.type = 'softmax';

% Load detections

data = load_detections(3);
img_o = data.img;
img = single(rgb2gray(img_o));
detec = data.detection.detection;

result_map_class_1 = zeros(500, 500);

for i = 1 : size(detec)
    c_p = detec(i, 1:2);
    c_x = int16(c_p(1, 1));
    c_y = int16(c_p(1, 2));
    
    patch = imcrop(img, [int16(c_x - 13.5), int16(c_y - 13.5), 26, 26]);
    patch_ = imresize(patch, net.meta.inputSize(1:2));
    patch_ = patch_ - net.meta.normalization.averageImage;
    patch_res = vl_simplenn(net, patch_);
    scores = squeeze(gather(patch_res(end).x));
    [bestScore, best] = max(scores);
    result_map_class_1(c_y,c_x) = result_map_class_1(c_y,c_x) + best;
end


figure, imshow(img_o);
hold on;
for y = 1 : 500
    for x = 1 : 500
        if result_map_class_1(y, x) == 1
            scatter(x, y, 'r');
            rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'r');
        elseif result_map_class_1(y, x) == 2
            scatter(x, y, 'g');
            rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'g');
        elseif result_map_class_1(y, x) == 3
            scatter(x, y, 'b');
            rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'b');
        elseif result_map_class_1(y, x) == 4
            scatter(x, y, 'y');
            rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'y');
        end
    end
end
hold off;


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