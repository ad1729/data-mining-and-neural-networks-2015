%% 2.1 Santa Fe time series prediction

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

%% Training a network
lags = 100; % 50 72 84
neurons = 20; % 10
train_alg = 'trainscg'; %trainscg, trainrp, trainbfg

train_data = tonndata(santafe, false, false);
test_data = tonndata(santafe_pred, false, false);

% setting up the training and validation sets
% net.divideFcn = 'divideind';
% net.divideParam = struct('trainInd', x_train, 'valInd', x_val , ...
% 'testInd', []);

% fitting the model
net = narnet(1:lags, neurons, 'closed',  train_alg); %'open' 'closed'

net.performParam.regularization = 0.000001;
net.trainParam.epochs = 2000;
net.performFcn = 'mae';  % 'mse', 'mae' use help nnperformance

[Xs,Xi,Ai,Ts] = preparets(net,{},{},train_data);
net = train(net,Xs,Ts,Xi,Ai);
%view(net)
y = net(Xs, Xi, Ai);

% calculating y_hat
Y_hat = nan(100+lags, 1); % creating an empty row vector
Y_hat = tonndata(Y_hat, false, false);
Y_hat(1:lags) = train_data((end-(lags-1)):end);
[xc, xic, aic, tc] = preparets(net, {}, {}, Y_hat);
Y_hat = fromnndata(net(xc, xic, aic), true, false, false);

figure;
plot(santafe_pred, 'r-');
hold on;
plot(Y_hat, 'g-');
hold off;
xlabel('Index');
title('Santa Fe Laser Prediction');
legend('Test Set', 'Approximated Function');

print('\home\ad\Desktop\images\santafe_pred', '-dpng');



%% 2.2 alphabet recognition (playing with the demo file inlcuded)
%appcr1 % using neural network for alphabet recognition
% We can make this more fun by taking the code from the help file for the
% demo and playing around with the neural network model

%% Noiseless case
[X,T] = prprob;
plotchar(X(:,4)); % number corresponds to the alphabet so 1-A, 4-D, etc

% feedforward neural network set up for pattern recognition with 25 hidden neurons
net1 = feedforwardnet(25);
view(net1)

net1.divideFcn = '';
net1 = train(net1,X,T,nnMATLAB);

%% Noisy case
numNoise = 30;
Xn = min(max(repmat(X,1,numNoise)+randn(35,26*numNoise)*0.2,0),1);
Tn = repmat(T,1,numNoise);

figure
plotchar(Xn(:,4)) % number corresponds to the alphabet so 1-A, 4-D, etc

net2 = feedforwardnet(25);
net2 = train(net2,Xn,Tn,nnMATLAB);

%% testing both networks
noiseLevels = 0:.05:1;
numLevels = length(noiseLevels);
percError1 = zeros(1,numLevels);
percError2 = zeros(1,numLevels);

for i = 1:numLevels
  Xtest = min(max(repmat(X,1,numNoise)+randn(35,26*numNoise)*noiseLevels(i),0),1);
  Y1 = net1(Xtest);
  percError1(i) = sum(sum(abs(Tn-compet(Y1))))/(26*numNoise*2);
  Y2 = net2(Xtest);
  percError2(i) = sum(sum(abs(Tn-compet(Y2))))/(26*numNoise*2);
end

figure
plot(noiseLevels,percError1*100,'--',noiseLevels,percError2*100);
title('Percentage of Recognition Errors');
xlabel('Noise Level');
ylabel('Errors');
legend('Network 1','Network 2','Location','NorthWest')

%% 2.3 Classification Problem (Pima Indian Diabetes)

inputs = Xnorm';
target = hardlim(Y)';

neurons = 15;

net = patternnet(neurons);

% divide the data into training, validation and test sets
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainFcn = 'trainlm'; % 'trainscg', 'trainlm', 'trainbfg', 'trainrp', 'traingd'
net.performParam.regularization = 1e-6;

[net, tr] = train(net, inputs, target);
%view(net)

plotperform(tr);

% evaluating the network on the test set
testX = inputs(:,tr.testInd);
testT = target(:, tr.testInd);

testY = net(testX);
testClass = testY > 0.5;

plotconfusion(testT, testY); % confusion matrix for the test set

% overall percentage of (in)correct classification on the test set
[c,cm] = confusion(testT,testY)

fprintf('Percentage Correct Classification   : %f%%\n', 100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n', 100*c);

plotroc(testT, testY) % ROC for the test set


