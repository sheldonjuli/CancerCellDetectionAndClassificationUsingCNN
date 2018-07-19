
% Run detection NN on every pixel's 27x27 surroundings and output a result
% Put Detection of crchistophenotypes_2016_04_28 in the same directory

clear variables;
close all;
clc;

% Load a model and upgrade it to MatConvNet current version.
trained_data = load('./detec-net-epoch-10.mat') ;
net = trained_data.net;
net = vl_simplenn_tidy(net) ;
net.layers{end}.type = 'softmax';

% % Obtain and preprocess an image.
% im = imread('2.jpg') ;
% im_ = single(im) ; % note: 255 range
% figure, imshow(im_);
% im_ = imresize(im_, net.meta.inputSize(1:2)) ;
% im_ = im_ - net.meta.normalization.averageImage ;


% Run the CNN.
% res = vl_simplenn(net, im_);
% 
% % Show the classification result.
% scores = squeeze(gather(res(end).x));
% [bestScore, best] = max(scores);

RAW_IMG_DIR = './Detection';
full_img = imread([RAW_IMG_DIR '/img3/img3.bmp']);
full_img = single(rgb2gray(full_img));
result_map = zeros(500,500);

tic;
for y = 1 : 500
    for x = 1 : 500
        fprintf('y : %d, x : %d \n', y, x);
        patch = imcrop(full_img, [int16(x - 13.5), int16(y - 13.5), 26, 26]);
        %figure, imshow(patch)
        patch_ = imresize(patch, net.meta.inputSize(1:2));
        patch_ = patch_ - net.meta.normalization.averageImage;
        patch_res = vl_simplenn(net, patch_);
        scores = squeeze(gather(patch_res(end).x));
        [bestScore, best] = max(scores);
        result_map(y,x) = result_map(y,x) + best;
    end
end
e = toc;

save('result_map_3.mat','result_map');

fprintf('Finished (took: %0.4f seconds)\n', e);
