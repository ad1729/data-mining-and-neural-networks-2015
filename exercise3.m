%% 3.1 Dimensionality Reduction By PCA
clear;
clc;

% the same code is run multiple times with 'trainlm' and 'trainbr', and
% with pn (all inputs) and pp (4 inputs)

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
[pp, pca_p] = processpca(pn, 'maxfrac', 0.001); %reduces data from 21 dimensions to 4 dimensions
[m, n] = size(pp)

%% Set indices for test, validation and training sets
Test_ix = 2:4:n;
Val_ix = 4:4:n;
Train_ix = [1:4:n 3:4:n];

%% Configure a network
% compare between LM and bayesian regularization (trainbr) (adjust the
% nodes in hidden layer appropriately)
net = fitnet(5, 'trainbr'); % trainlm and trainbr 

net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', Train_ix, ...
'valInd', Val_ix, ...
'testInd', Test_ix);
[net, tr] = train(net, pp, tn); % pn - original data; pp - reduced data

%% Get predictions on training and test
% pn - original data; pp - reduced data
Yhat_train = net(pp(:, Train_ix));
Yhat_test = net(pp(:, Test_ix));
perf_train = perform(net, tn(:, Train_ix), Yhat_train);
perf_test = perform(net, tn(:, Test_ix), Yhat_test);

%% 3.2 Automatic Relevance Detection
% In order to apply ARD, download the Netlab software from
% http://www.ncrg.aston.ac.uk/netlab/ or use the following link to
% get to the download's page directly: http://cl.ly/Rw0P

clear; % clear workspace
clc; % clear console

%% (a) run demo demard
demard

%% (b) run demo demev1
demev1

%% (c) UCI ionosphere data classification using MLP and ARD
% code in this section is taken from demard.m with appropriate
% modifications where necessary
load('/home/ad/Desktop/KUL Course Material/Data Mining And Neural Networks/Final exam/ionstart.mat', '-mat')
[inputs, std_input] = mapstd(Xnorm); % normalizing the input variables
targets = hardlim(Y); % converting from [-1,1] to [0,1] thus no need to find hyperparameter beta

train_ind = [1:6:351 2:6:351 4:6:351 5:6:351]; % 67% of the data used as training
test_ind = [3:6:351 6:6:351]; % 33% of the data used as a test set

%% Set up network parameters.
nin = 33;			% Number of inputs.
nhidden = 2;			% Number of hidden units.
nout = 1;			% Number of outputs.
aw1 = 0.01*ones(1, nin);	% First-layer ARD hyperparameters.
ab1 = 0.01;			% Hyperparameter for hidden unit biases.
aw2 = 0.01;			% Hyperparameter for second-layer weights.
ab2 = 0.01;			% Hyperparameter for output unit biases.
%beta = 50.0;			% Coefficient of data error.
% do not need to estimate beta since the likelihood function is taken to be
% the cross-entropy function

%% Create and initialize network.
prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'logistic', prior);%, beta); 
% logistic because binary classification problem

%% Set up vector of options for the optimiser.
nouter = 10;			% Number of outer loops
ninner = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 1000;		% Number of training cycles in inner loop. 

%% Train using scaled conjugate gradients, re-estimating alpha and beta.
for k = 1:nouter
  net = netopt(net, options, inputs(train_ind,:), targets(train_ind), 'scg');
  [net, gamma, logev] = evidence(net, inputs(train_ind,:), targets(train_ind), ninner);
  clc;
end

%% Selecting Parameters
i = 1:nin;
val = net.alpha(1:nin,:);
fprintf(1, '  alpha%i =  %8.5f\n', [i; val']);
%fprintf(1, '  beta  =  %8.5f\n', net.beta);
fprintf(1, '  gamma =  %8.5f\n\n', gamma);
fprintf(1, '  log-likelihood =  %8.5f\n\n', logev);

%% Display the weights for each of the inputs
%weight_table = table(index, net.w1(:,1)', net.w1(:,2)');
disp('This is confirmed by looking at the corresponding weight values:')
disp(' ');
fprintf(1, '    x%i:    %8.5f    %8.5f\n', [i; net.w1']);
disp(' ');

%% Predictions from the model and comparing with the test set
[pred, z] = mlpfwd(net, inputs(test_ind, :));

%% Plot confusion matrix and ROC curves for the network with all inputs
plotconfusion(targets(test_ind)', pred');
print('\home\ad\Desktop\images\ion_pred_full_confusion', '-dpng');

plotroc(targets(test_ind)', pred');
print('\home\ad\Desktop\images\ion_pred_full_roc', '-dpng');

%log-likelihood =  -9646.29802

%% Training the network using a smaller number of inputs
clear;
clc;

load('/home/ad/Desktop/KUL Course Material/Data Mining And Neural Networks/Final exam/ionstart.mat', '-mat')
[inputs, std_input] = mapstd(Xnorm); % normalizing the input variables
targets = hardlim(Y); % converting from [-1,1] to [0,1] thus no need to find hyperparameter beta

train_ind = [1:6:351 2:6:351 4:6:351 5:6:351]; % 67% of the data used as training
test_ind = [3:6:351 6:6:351]; % 33% of the data used as a test set

reduced_inputs = [8 19 21 22 26 33];

% same as for the case with all inputs
nin = length(reduced_inputs);			% Number of inputs.
nhidden = 2;			% Number of hidden units.
nout = 1;			% Number of outputs.
aw1 = 0.01*ones(1, nin);	% First-layer ARD hyperparameters.
ab1 = 0.01;			% Hyperparameter for hidden unit biases.
aw2 = 0.01;			% Hyperparameter for second-layer weights.
ab2 = 0.01;			% Hyperparameter for output unit biases.

prior = mlpprior(nin, nhidden, nout, aw1, ab1, aw2, ab2);
net = mlp(nin, nhidden, nout, 'logistic', prior);%, beta); 

nouter = 10;			% Number of outer loops
ninner = 10;		        % Number of inner loops
options = zeros(1,18);		% Default options vector.
options(1) = 1;			% This provides display of error values.
options(2) = 1.0e-7;	% This ensures that convergence must occur
options(3) = 1.0e-7;
options(14) = 1000;		% Number of training cycles in inner loop. 

% include selected inputs instead of all in the next two lines
for k = 1:nouter
  net = netopt(net, options, inputs(train_ind, reduced_inputs), targets(train_ind), 'scg');
  [net, gamma, logev] = evidence(net, inputs(train_ind, reduced_inputs), targets(train_ind), ninner);
  clc;
end

i = 1:nin;
val = net.alpha(1:nin,:);
fprintf(1, '  alpha%i =  %8.5f\n', [i; val']);
%fprintf(1, '  beta  =  %8.5f\n', net.beta);
fprintf(1, '  gamma =  %8.5f\n\n', gamma);
fprintf(1, '  log-likelihood =  %8.5f\n\n', logev);

% Predictions from the model and comparing with the test set (using only the relevant inputs)
[pred, z] = mlpfwd(net, inputs(test_ind, reduced_inputs));

% Plot confusion matrix and ROC curves
plotconfusion(targets(test_ind)', pred');
print('\home\ad\Desktop\images\ion_pred_reduced_confusion', '-dpng');

plotroc(targets(test_ind)', pred');
print('\home\ad\Desktop\images\ion_pred_reduced_roc', '-dpng');

%log-likelihood =  -643.50711