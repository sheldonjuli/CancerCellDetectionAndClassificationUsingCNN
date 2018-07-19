
% Evaluate classification NN accuracy using images in test

clear variables;
close all;
clc;

result1 = runTest([], 1, 6001, 6100);
result2 = runTest(result1, 2, 2001, 2100);
result3 = runTest(result2, 3, 1901, 2000);
result4 = runTest(result3, 4, 1001, 1100);

save('test_classification_accuracy_result', 'result4');

function result = runTest(prev_result, label, start_id, end_id)

    % Load a model and upgrade it to MatConvNet current version.
    trained_data = load('./class-net-epoch-10.mat') ;
    net = trained_data.net;
    net = vl_simplenn_tidy(net) ;
    net.layers{end}.type = 'softmax';
    
    result = prev_result;
    for i = start_id : end_id
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

