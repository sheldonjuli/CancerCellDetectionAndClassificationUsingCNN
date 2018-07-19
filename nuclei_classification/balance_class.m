
% Generates permutations of given images to balance classes

IN_DIR = '../../data/nuclei-class-dataset/train/2';
OUT_DIR = '../../data/nuclei-class-dataset/train/2';

generatePermu(IN_DIR, OUT_DIR, 1, 2934)

IN_DIR = '../../data/nuclei-class-dataset/train/4';
OUT_DIR = '../../data/nuclei-class-dataset/train/4';

generatePermu(IN_DIR, OUT_DIR, 1, 1517)


IN_DIR = '../../data/nuclei-class-dataset/val/4';
OUT_DIR = '../../data/nuclei-class-dataset/val/4';

generatePermu(IN_DIR, OUT_DIR, 1, 314)

function generatePermu(inDir, outDir, inId, outId)
    
    temp_id = outId + 1;
    for i = inId : outId
        im = imread(sprintf('%s/%d.jpg', inDir, i));
        imwrite(imrotate(im, 90), sprintf('%s/%d.jpg', outDir, temp_id), 'jpg');
        temp_id = temp_id + 1;
        imwrite(imrotate(im, 180), sprintf('%s/%d.jpg', outDir, temp_id), 'jpg');
        temp_id = temp_id + 1;
        imwrite(imrotate(im, 270), sprintf('%s/%d.jpg', outDir, temp_id), 'jpg');
        temp_id = temp_id + 1;
    end
end
