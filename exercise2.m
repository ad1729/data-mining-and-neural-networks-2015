%% 2.1 Santa Fe time series prediction
clear;
clc;

% reading in the training set
fid = fopen('/home/ad/Desktop/KUL Course Material/Data Mining And Neural Networks/MATLAB/Exercises/lasertrain.dat','r');
datacell = textscan(fid, '%f', 'Delimiter','\n', 'CollectOutput', 1);
fclose(fid);
A.data = datacell{1};
santafe = A.data;

% reading in the prediction set
fid = fopen('/home/ad/Desktop/KUL Course Material/Data Mining And Neural Networks/MATLAB/Exercises/laserpred.dat','r');
datacell = textscan(fid, '%f', 'Delimiter','\n', 'CollectOutput', 1);
fclose(fid);
B.data = datacell{1};
santafe_pred = B.data;

% santafe = tonndata('/home/ad/Desktop/KUL Course Material/Data Mining And Neural Networks/MATLAB/Exercises/lasertrain.dat', false, false)
% santafe_pred = tonndata('/home/ad/Desktop/KUL Course Material/Data Mining And Neural Networks/MATLAB/Exercises/laserpred.dat', false, false)

%% plotting both side by side
figure

subplot(1,2,1)
plot(santafe, '-');
title('Training Set');

subplot(1,2,2)
plot(santafe_pred, 'r-');
title('Test Set');

print('\home\ad\Desktop\images\santafe', '-dpng');

figure;
subplot(2,1,1, 'align')
autocorr(mapminmax(santafe), 500) %acf
subplot(2,1,2, 'align')
parcorr(mapminmax(santafe), 500) %pacf
print('\home\ad\Desktop\images\santafe_acf', '-dpng');

%% Setting up the network
lags = 80; % 16, 25, 40, 60
%neurons = round(lags/2); % rule of thumb
neurons = 30;
train_alg = 'trainscg'; % trainscg, trainbfg, trainlm, trainbr

% converting into a form which can be used by the narnet function
train_data = con2seq(santafe');
test_data = con2seq(santafe_pred');

% fitting the model, model to be trained in feedforward mode
net = narnet(1:lags, neurons, 'open', train_alg); %'open' (default)- feedforward 'closed'- recurrent

net.performParam.regularization = 1e-6;
net.trainParam.epochs = 1000; % 2000
net.performFcn = 'mse';  % 'mse', 'mae'; info: use help nnperformance

net.divideFcn = 'divideblock'; % not splitting the data into train/validation because time series
% but using narnet has net.divideMode = 'time' so it's cool?
% setting up the training and validation sets

%[trainInd,valInd] = divideint(1000, 0.8, 0.2);
% splitting the set into 70% for training and 30% for validation
% trainInd = [1:70, 101:170, 201:270, 301:370, 401:470, 501:570, 601:670, 701:770, 801:870, 901:970];
% valInd = [71:100, 171:200, 271:300, 371:400, 471:500, 571:600, 671:700, 771:800, 871:900, 971:1000];
% net.divideFcn = 'divideind';
% net.divideParam = struct('trainInd', trainInd, 'valInd', valInd, 'testInd', []);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 30/100;
net.divideParam.testRatio = 0/100;

%% Training the network
[Xs,Xi,Ai,Ts] = preparets(net,{},{},train_data); 
[net, tr] = train(net,Xs,Ts,Xi,Ai);

Y = net(Xs,Xi,Ai); 
perf = perform(net,Ts,Y)

train_errors = cell2mat(gsubtract(Ts,Y));

% closing the loop and doing 100 multi-step ahead predictions
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
Y_hat = [train_data((1000-lags+1):1000) num2cell(nan(100,1)')];

[xc,xic,aic,tc] = preparets(netc,{},{},Y_hat);
Y_hat = netc(xc,xic,aic);

% calculating the residuals and the MAE
test_residuals = gsubtract(test_data,Y_hat);
mae = 0.01 * sum(abs(cell2mat(test_residuals))) % 0.01 = 1/100; formula uses 1/N

%% Saving the predicted function
figure;

% subplot(1,3,1, 'align')
% autocorr(train_errors, 100) %acf
% 
% subplot(1,3,2, 'align')
% parcorr(train_errors, 100) %pacf
% 
% subplot(1,3,3, 'align')
plot(santafe_pred, '-+');
hold on;
plot(cell2mat(Y_hat), 'r-*');
hold off;
xlabel('Index');
title('Santa Fe Laser Prediction');
legend('Test Set', 'Approximated Function');

%print('\home\ad\Desktop\images\santafe_pred60', '-dpng');
%print('\home\ad\Desktop\images\manual_lag60', '-dpng');

%% 2.2 alphabet recognition (playing with the demo file inlcuded)
% appcr1 % using neural network for alphabet recognition

%% 2.3 Classification Problem (Pima Indian Diabetes)
clear;
clc;

load('/home/ad/Desktop/KUL Course Material/Data Mining And Neural Networks/Final exam/pidstart.mat', '-mat')

%% Training the network
[inputs, std_input] = mapstd(Xnorm'); % normalizing the input variables
target = hardlim(Y)'; % converting from [-1,1] to [0,1]

neurons = 10; % 2, 5, 8, 10 (check ROC print statement below as well!)

net = patternnet(neurons);

% divide the data into training, validation and test sets
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;

[net, tr] = train(net, inputs, target);
perf = tr.best_vperf
%view(net)

%% evaluating the network on the test set
testX = inputs(:, tr.testInd);
testT = target(:, tr.testInd);

testY = net(testX);
testClass = testY > 0.5;

plotroc(testT, testY) % ROC for the test set
print('\home\ad\Desktop\images\pima_roc_10', '-dpng');

plotconfusion(testT, testY) % confusion matrix for the test set
print('\home\ad\Desktop\images\pima_confusion_10', '-dpng');

% overall percentage of (in)correct classification on the test set
[c,cm] = confusion(testT,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);