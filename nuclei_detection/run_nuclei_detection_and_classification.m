
% Classify nuclei from detection points returned by NN and plot the result

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
detec_map = data.detection.trimmed_map;

figure, imshow(img_o);
hold on;
for y = 1 : 500
    for x = 1 : 500
        if detec_map(y, x) == 2
            patch = imcrop(img, [int16(x - 13.5), int16(y - 13.5), 26, 26]);
            patch_ = imresize(patch, net.meta.inputSize(1:2));
            patch_ = patch_ - net.meta.normalization.averageImage;
            patch_res = vl_simplenn(net, patch_);
            scores = squeeze(gather(patch_res(end).x));
            [bestScore, best] = max(scores);

            if best == 1
                scatter(x, y, 'r');
                rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'r');
            elseif best == 2
                scatter(x, y, 'g');
                rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'g');
            elseif best == 3
                scatter(x, y, 'b');
                rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'b');
            elseif best == 4
                scatter(x, y, 'y');
                rectangle('Position',[x-13 y-13 27 27],'EdgeColor', 'y');
            end
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
        data.detection = load(sprintf('trimmed_map_%d.mat', img_id));
    end
end