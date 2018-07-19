CROPPED_RESULT_DIR = '../data/nuclei-dataset/class';

% % % train
%%%%%%%%%%%%%%%%
dir = sprintf('%s/train', CROPPED_RESULT_DIR);
dim = 27*27*3;
train_size = 1500;
train_mat = zeros(6000, dim);
train_label = ones(6000, 1);
train_label(1501:3000) = 2;
train_label(3001:4500) = 3;
train_label(4501:6000) = 4;
for j = 1 : 4
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
save(char(sprintf('%s/../svm/svmclasscolmodel', CROPPED_RESULT_DIR)),'model');

% pause;

% % % test
model = load(sprintf('%s/../svm/svmclasscolmodel.mat', CROPPED_RESULT_DIR));
model = model.model;
%%%%%%%%%%%%%%%%%%
dir = sprintf('%s/testall70', CROPPED_RESULT_DIR);
test_label = load(sprintf('%s/classifications.mat', dir));
test_label = test_label.all_class;
dim = 27*27*3;
test_mat = zeros(size(test_label,1), dim);
for i = 1 : size(test_label,1)
    file = fullfile(dir, sprintf('/%d.jpg', i));
    img = double((imread(file)));
    norm = img(:) / 255;
    test_mat(i,:) = norm(:);
end
%%%%%%%%%%%%%%%%%%%%%%
fprintf("prepared for svmpredict!\n");
[predictions, accuracy, probs] = svmpredict(test_label, test_mat, model, '-b 1');
fprintf("svmpredict'd!\n");
result = [predictions test_label];
save(char(sprintf('%s/../svm/svmclasscolresult', CROPPED_RESULT_DIR)),'result');

%%%% calculate scores
scores = zeros(5,3);
for i = 1 : 4
    pos = result(result(:,2) == i);
    neg = result(result(:,2) ~= i);
    true_p = size(result(pos(:,1) == i));
    false_p = size(result(pos(:,1) ~= i));
    true_n = size(result(neg(:,1) ~= i));
    false_n = size(result(neg(:,1) == i));
    precision = true_p/(true_p+false_p);
    recall = true_p/(true_p+false_n);
    f1 = 2*((precision*recall)/(precision+recall));
    scores(i,:) = [precision, recall, f1];
end
%%% averages:
scores(5,:) = [mean(scores(:,1)) mean(scores(:,2)) mean(scores(:,3))];
scores
save(char(sprintf('%s/../svm/svmclasscolscores', CROPPED_RESULT_DIR)),'scores');

