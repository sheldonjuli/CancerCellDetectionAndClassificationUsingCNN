CROPPED_RESULT_DIR = '../data/nuclei-dataset/detec';

% % % train
%%%%%%%%%%%%%%%%
dir = sprintf('%s/train', CROPPED_RESULT_DIR);
dim = 27*27*3;
train_size = 1500;
train_mat = zeros(3000, dim);
train_label = ones(3000, 1);
train_label(1501:3000) = 2;
for j = 1 : 2
    for i = 1 : train_size
        file = fullfile(dir, sprintf('/%d/%d.jpg', j,i));
        img = double((imread(file)));
        norm = img(:) / 255;
        train_mat((train_size*(j-1))+i,:) = norm(:);
    end
end
%%%%%%%%%%%%%%
fprintf("prepared for svmtrain!\n");
model = svmtrain(train_label, train_mat, '-m 1000 -b 1 -c 32.0 -g 0.0078125');
fprintf("svmtrain'd!\n");
save(char(sprintf('%s/../svm/svmdeteccolmodel', CROPPED_RESULT_DIR)),'model');

pause;

% % % test
model = load(sprintf('%s/../svm/svmdeteccolmodel.mat', CROPPED_RESULT_DIR));
model = model.model;
%%%%%%%%%%%%%%%%%%
wind_size = 27;
dim = 27*27*3;
step = 5;
im_index = 65; % choosing a test image
subcount = 0;
% setup
dir = sprintf('%s/test', CROPPED_RESULT_DIR);
winds = floor(474/step)*floor(474/step); % number of windows
detecs = load(sprintf('%s/img%d/classifications.mat', dir,im_index));
detecs = detecs.classif;
file = fullfile(dir, sprintf('/img%d/img%d.bmp', im_index, im_index));
img = double((imread(file)));
detecs_r = round(detecs);
norm = img / 255; % libSVM has far better results with scaled values.
test_label = zeros(winds, 1);
test_mat = zeros(winds, dim);
for i = 1 : step : 474 % move window 5 pixels each iteration
for k = 1 : step : 474
    subcount = subcount + 1;
    % make wind with (i,k) as corner
    wind = imcrop(norm, [i, k, 26, 26]); % make patches of size 27x27
    % find ground truths for this window
    filt = detecs_r(:,1)>(i-7) & detecs_r(:,1)<(i+26+7) & detecs_r(:,2)>(k-7) & detecs_r(:,2)<(k+26+7);
    if size(detecs_r(filt),1) > 0
        % a nucleus should be detected here
        test_label(subcount) = 2;
    else
        test_label(subcount) = 1;
    end
    % append test case
    test_mat(subcount,:) = wind(:);
end
% printed progress
if mod(i,200) == 0
    fprintf("image %d: %d columns scanned\n", im_index, i);
end
end
fprintf("prepared for svmpredict!\n");
[predictions, accuracy, probs] = svmpredict(test_label, test_mat, model, '-b 1');
fprintf("svmpredict'd!\n");
result = [predictions test_label];
save(char(sprintf('%s/../svm/svmdeteccolresult', CROPPED_RESULT_DIR)),'result');

%%%% calculate scores
%%% a positive is when a nucleus is detected
pos = result(result(:,2) == 2);
neg = result(result(:,2) == 1);
true_p = size(result(pos(:,1) == 2));
false_p = size(result(pos(:,1) == 1));
true_n = size(result(neg(:,1) == 1));
false_n = size(result(neg(:,1) == 2));

precision = true_p/(true_p+false_p);
recall = true_p/(true_p+false_n);
f1 = 2*((precision*recall)/(precision+recall));

scores = [precision, recall, f1];
scores
save(char(sprintf('%s/../svm/svmdeteccolscores', CROPPED_RESULT_DIR)),'scores');


