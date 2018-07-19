CROPPED_RESULT_DIR = '../data/nuclei-dataset/class';

model = load(sprintf('%s/../svm/svmclasscolmodel.mat', CROPPED_RESULT_DIR));
net = load(sprintf('%s/../svm/net-epoch-10-class.mat', CROPPED_RESULT_DIR));
net_detec = load(sprintf('%s/../svm/net-epoch-10-detec.mat', CROPPED_RESULT_DIR));
model = model.model;
net = net.net;
net.layers{end}.type = 'softmax';
net_detec = net_detec.net;
net_detec.layers{end}.type = 'softmax';

%%%%%%%%%%%%%%%%%%
%%%% classification with NEP
tic;
NEP_d = 4; % same as paper
img_index = 65; %65 good
dim = 27*27*3;
% setup
dir = sprintf('%s/test', CROPPED_RESULT_DIR);
test_label = load(sprintf('%s/img%d/classifications.mat', dir,img_index));
test_label = test_label.classif;
result = [];
file = fullfile(dir, sprintf('/img%d/img%d.bmp', img_index,img_index));
% SVM
img = double(rgb2gray(imread(file)));
% Neural Net
% img = single(rgb2gray(imread(file)));
for i = 1 : size(test_label,1)
    c_x = test_label(i,1);
    c_y = test_label(i,2);
    if c_x-13 > 0 && c_y-13 > 0 && c_x+13 < 500 && c_y+13 < 500
    w = 1/((NEP_d+1)^2);
    % Neural Net
%     p = [];
    % SVM speedup
    p = zeros((NEP_d+1)^2, 4);
    subcount = 1;
    for k = 1 : NEP_d+1
    for l = 1 : NEP_d+1
        curx = c_x + k;
        cury = c_y + l;
        if curx-13 > 0 && cury-13 > 0 && curx+13 < 500 && cury+13 < 500
            wind = imcrop(img, [curx-13, cury-13, 26, 26]); % make patches of size 27x27
            %%% libSVM
            norm = wind / 255;
            test_img = norm(:).';
            [spred, sacc, sprob] = svmpredict(test_label(i,3), test_img, model, '-b 1 -q');
            den = sum(exp(sprob));
            p(subcount,:) = (exp(sprob) / den) * w;
            subcount = subcount + 1;
            %%% MatConvNet
%             nndres = vl_simplenn(net_detec, wind);
%             dprob = squeeze(gather(nndres(end).x)).';
%             if dprob(2) > dprob(1)
%                 nnres = vl_simplenn(net, wind);
%                 sprob = squeeze(gather(nnres(end).x)).';
%                 den = sum((sprob));
%                 p = [p ; ((sprob) / den) * w];
%                 subcount = subcount + 1;
%             end
        end
    end
    end
    [nep_m, class] = max(sum(p));
%     % SVM classes are shifted by one
    class = class + 1;
    if class > 4
        class = 1;
    end
    result = [result; class, test_label(i,3)];
    end
    
    if mod(i,200) == 0
        fprintf('%d\n',i);
    end
end
e = toc;

fprintf('finished image %d. (took: %0.4f seconds)\n', img_index, e);

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
save(char(sprintf('%s/../svm/svmnepclassscores', CROPPED_RESULT_DIR)),'scores');
