%% 3.1 Dimensionality Reduction By PCA

%% Load the cholesterol data (comes with matlab)    
% This will create a 21x264 choInputs matrix of 264 input patterns
% and a 3x264 matrix choTargets of output patterns
%doc cho_dataset % dataset details
load cho_dataset

%% Standardize the variables

% doc mapstd;
[pn, std_p] = mapstd(choInputs);
[tn, std_t] = mapstd(choTargets);

%% PCA

%doc processpca;
[pp, pca_p] = processpca(pn, 'maxfrac', 0.001);
[m, n] = size(pp)

%% Set indices for test, validation and training sets
Test_ix = 2:4:n;
Val_ix = 4:4:n;
Train_ix = [1:4:n 3:4:n];

%% Configure a network
net = fitnet(5, 'trainlm'); % compare with bayesian regularization (trainbr)
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', Train_ix, ...
'valInd', Val_ix, ...
'testInd', Test_ix);
[net, tr] = train(net, pn, tn);

%% Get predictions on training and test
Yhat_train = net(pn(:, Train_ix));
Yhat_test = net(pn(:, Test_ix));


