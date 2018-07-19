
% Creates imdb and train classification NN

function [net, info] = cnn_nuclei(varargin)

run([fileparts(mfilename('fullpath')) '/../../matlab/vl_setupnn.m']) ;

% Parameter defaults.
opts.train.batchSize = 400 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.01 ;
opts.train.expDir = [vl_rootnn '/data/nuclei-class'];
opts.dataDir = [vl_rootnn '/data/nuclei-class-dataset'] ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = [opts.train.expDir '/imdb.mat'] ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

% Print an error message if they don't exist
if ~exist(opts.dataDir, 'dir')
    fprintf('Data directory does not exist.');
end

% Create image database (imdb struct). It can be cached to a file for speed
if exist(opts.imdbPath, 'file')
  disp('Reloading image database...')
  imdb = load(opts.imdbPath) ;
else
  disp('Creating image database...')
  imdb = getImdb(opts.dataDir) ;
  mkdir(fileparts(opts.imdbPath)) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% Create network (see HELP VL_SIMPLENN)
f = 1/100 ;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(4,4,1,36, 'single'), zeros(1, 36, 'single')}});
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2);
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(3,3,36,48, 'single'),zeros(1,48,'single')}});
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'max', ...
                           'pool', [2 2], ...
                           'stride', 2);
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(5,5,48,512, 'single'),  zeros(1,512,'single')}});
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,512,512, 'single'),  zeros(1,512,'single')}});
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{f*randn(1,1,512,4, 'single'),  zeros(1,4,'single')}});
net.layers{end+1} = struct('type', 'softmaxloss');

% Meta parameters
net.meta.inputSize = [27 27 1];
net.meta.trainOpts.learningRate = 0.01;
net.meta.trainOpts.numEpochs = 20;
net.meta.trainOpts.batchSize = 400;
net.meta.normalization.averageImage =imdb.data_mean;

% Fill in any values we didn't specify explicitly
net = vl_simplenn_tidy(net) ;

use_gpu = ~isempty(opts.train.gpus) ;

% Start training
[net, stats] = cnn_train(net, imdb, @(imdb, batch) getBatch(imdb, batch, use_gpu), ...
  'train', find(imdb.set == 1), 'val', find(imdb.set == 2), opts.train) ;

% Visualize the learned filters
figure(3) ; vl_tshow(net.layers{1}.weights{1}) ; title('Conv1 filters, layer 1');
figure(4) ; vl_tshow(net.layers{3}.weights{1}) ; title('Conv2 filters, layer 1');
figure(5) ; vl_tshow(net.layers{5}.weights{1}) ; title('Conv3 filters, layer 1');
figure(6) ; vl_tshow(net.layers{6}.weights{1}) ; title('Conv4 filters, layer 6');
figure(7) ; vl_tshow(net.layers{7}.weights{1}) ; title('Conv5 filters, layer 7');

% --------------------------------------------------------------------
function [images, labels] = getBatch(imdb, batch, use_gpu)
% --------------------------------------------------------------------
% This is where we return a given set of images (and their labels) from
% our imdb structure.
% If the dataset was too large to fit in memory, getBatch could load images
% from disk instead (with indexes given in 'batch').

images = imdb.images(:,:,:,batch) ;
labels = imdb.labels(batch) ;
if use_gpu
	images = gpuArray(images) ;
end

% --------------------------------------------------------------------
function imdb = getImdb(dataDir)
% --------------------------------------------------------------------
% Initialize the imdb structure (image database).
% Note the fields are arbitrary: only your getBatch needs to understand it.
% The field imdb.set is used to distinguish between the training and
% validation sets, and is only used in the above call to cnn_train.

% The sets, and number of samples per label in each set
sets = {'train', 'val'} ;
numSamples = [3000, 300] ;

% Preallocate memory
totalSamples = 13200 ;  % 4000 * 4 + 700 * 4
images = zeros(27, 27, 1, totalSamples, 'single') ;
labels = zeros(totalSamples, 1) ;
set = ones(totalSamples, 1);

% Read all samples
sample = 1 ;
for s = 1:2  % Iterate sets
    for label = 1:4  % Iterate labels
        for i = 1:numSamples(s)  % Iterate samples
          % Read image
          im = imread(sprintf('%s/%s/%i/%d.jpg', dataDir, sets{s}, label, i)) ;
          % imshow(im);
          % Store it, along with label and train/val set information
          images(:,:,:,sample) = single(im) ;
          labels(sample) = label ;
          set(sample) = s ;
          sample = sample + 1 ;
        end
    end
end

% Remove mean over whole dataset
dataMean=mean(images,4);
images = bsxfun(@minus, images, dataMean) ;

% Store results in the imdb struct
imdb.images = images ;
imdb.labels = labels ;
imdb.set = set ;
imdb.data_mean = dataMean;


