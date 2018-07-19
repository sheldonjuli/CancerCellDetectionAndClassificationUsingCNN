
% Test the accuracy of NN on images in test

clear variables;
close all;
clc;

result1 = runTest([], 1);
result2 = runTest(result1, 2);

save('test_detection_accuracy_result', 'result2');

function result = runTest(prev_result, label)

    % Load a model and upgrade it to MatConvNet current version.
    trained_data = load('./detec-net-epoch-10.mat') ;
    net = trained_data.net;
    net = vl_simplenn_tidy(net) ;
    net.layers{end}.type = 'softmax';
    
    result = prev_result;
    for i = 3001 : 3200
        % Obtain and preprocess an image.
        im = imread((sprintf('test/%d/%d.jpg', label, i)));
        im_ = single(im) ; % note: 255 range
        im_ = imresize(im_, net.meta.inputSize(1:2)) ;
        im_ = im_ - net.meta.normalization.averageImage ;

        % Run the CNN.
        res = vl_simplenn(net, im_);
        % Show the classification result.
        scores = squeeze(gather(res(end).x));
        [bestScore, best] = max(scores);
        result = [result ; best, label];
    end
end

