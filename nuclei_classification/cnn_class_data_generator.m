
% Put Classification of crchistophenotypes_2016_04_28 in the same directory

CROPPED_RESULT_DIR = '../../data/nuclei-class-dataset';
count = 0;
train_size = 80; % val size will just be 100-train_size
%%%%%%%%%%%%%%%%
tic;
epi_class = [];
mkdir(sprintf('%s/train/1', CROPPED_RESULT_DIR));
inflam_class = [];
mkdir(sprintf('%s/train/2', CROPPED_RESULT_DIR));
fib_class = [];
mkdir(sprintf('%s/train/3', CROPPED_RESULT_DIR));
other_class = [];
mkdir(sprintf('%s/train/4', CROPPED_RESULT_DIR));

for j = 1 : train_size
    fprintf("Collected %d. Collecting samples from img%d.\n", count, j);
    data = load_detections(j);
    img = data.img;
    classif = data.detection;
    for i = 1 : size(classif)
        c_p = classif(i, 1:3);
        c_x = c_p(1, 1);
        c_y = c_p(1, 2);
        
        if c_x >= 14 && c_x <= 486 && c_y >= 14 && c_y <= 486
            wind = imcrop(img, [int16(c_x - 13.5), int16(c_y - 13.5), 26, 26]); % make patches of size 27x27
            if c_p(1,3) == 1
                class = 'epithelial';
                count = size(epi_class, 1) + 1;
                epi_class = [epi_class ; c_p(1,3)];
            elseif c_p(1,3) == 2
                class = 'inflammatory';
                count = size(inflam_class, 1) + 1;
                inflam_class = [inflam_class ; c_p(1,3)];
            elseif c_p(1,3) == 3
                class = 'fibroblast';
                count = size(fib_class, 1) + 1;
                fib_class = [fib_class ; c_p(1,3)];
            elseif c_p(1,3) == 4
                class = 'other';
                count = size(other_class, 1) + 1;
                other_class = [other_class ; c_p(1,3)];
            end
            img_name = sprintf('%s/train/%d/%d.jpg', CROPPED_RESULT_DIR, c_p(1,3), count);
            imwrite(rgb2gray(wind), img_name, 'jpg');
        end
    end

end
save(char(sprintf('%s/train/1', CROPPED_RESULT_DIR)),'epi_class');
save(char(sprintf('%s/train/2', CROPPED_RESULT_DIR)),'inflam_class');
save(char(sprintf('%s/train/3', CROPPED_RESULT_DIR)),'fib_class');
save(char(sprintf('%s/train/4', CROPPED_RESULT_DIR)),'other_class');

e = toc;

fprintf('Collected %d. (took: %0.4f seconds)\n', count, e);
%%%%%%%%%%%%%%%%%%%
tic;
epi_class = [];
mkdir(sprintf('%s/val/1', CROPPED_RESULT_DIR));
inflam_class = [];
mkdir(sprintf('%s/val/2', CROPPED_RESULT_DIR));
fib_class = [];
mkdir(sprintf('%s/val/3', CROPPED_RESULT_DIR));
other_class = [];
mkdir(sprintf('%s/val/4', CROPPED_RESULT_DIR));
for j = train_size+1 : 100 % val portion
    fprintf("Collected %d. Collecting samples from img%d.\n", count, j);
    data = load_detections(j);
    img = data.img;
    classif = data.detection;
    for i = 1 : size(classif)
        c_p = classif(i, 1:3);
        c_x = c_p(1, 1);
        c_y = c_p(1, 2);
        
        if c_x >= 14 && c_x <= 486 && c_y >= 14 && c_y <= 486
            wind = imcrop(img, [int16(c_x - 13.5), int16(c_y - 13.5), 26, 26]); % make patches of size 27x27
            if c_p(1,3) == 1
                class = 'epithelial';
                count = size(epi_class, 1) + 1;
                epi_class = [epi_class ; c_p(1,3)];
            elseif c_p(1,3) == 2
                class = 'inflammatory';
                count = size(inflam_class, 1) + 1;
                inflam_class = [inflam_class ; c_p(1,3)];
            elseif c_p(1,3) == 3
                class = 'fibroblast';
                count = size(fib_class, 1) + 1;
                fib_class = [fib_class ; c_p(1,3)];
            elseif c_p(1,3) == 4
                class = 'other';
                count = size(other_class, 1) + 1;
                other_class = [other_class ; c_p(1,3)];
            end
            img_name = sprintf('%s/val/%d/%d.jpg', CROPPED_RESULT_DIR, c_p(1,3), count);
            imwrite(rgb2gray(wind), img_name, 'jpg');
        end
    end

end
save(char(sprintf('%s/val/1', CROPPED_RESULT_DIR)),'epi_class');
save(char(sprintf('%s/val/2', CROPPED_RESULT_DIR)),'inflam_class');
save(char(sprintf('%s/val/3', CROPPED_RESULT_DIR)),'fib_class');
save(char(sprintf('%s/val/4', CROPPED_RESULT_DIR)),'other_class');

e = toc;

fprintf('Collected %d. (took: %0.4f seconds)\n', count, e);
%%%%%%%%%%%%%%

function data = load_detections(img_id)
    RAW_IMG_DIR = './Classification';
    img_dir = sprintf('%s/img%d', RAW_IMG_DIR, img_id);
    files = dir(fullfile(img_dir, sprintf('/img%d*', img_id)));
    if isempty(files)
        fprintf('file doesn''t exist!\n');
    else
        data.img = imread(fullfile(img_dir, files(1).name));
        epi = load(fullfile(img_dir, files(2).name));
        epi = [epi.detection, ones(size(epi.detection, 1), 1)];
        fib = load(fullfile(img_dir, files(3).name));
        fib = [fib.detection, 2*ones(size(fib.detection, 1), 1)];
        infla = load(fullfile(img_dir, files(4).name));
        infla = [infla.detection, 3*ones(size(infla.detection, 1), 1)];
        others = load(fullfile(img_dir, files(5).name));
        others = [others.detection, 4*ones(size(others.detection, 1), 1)];
        data.detection = [epi ;
                          fib ;
                          infla ;
                          others];
    end
end