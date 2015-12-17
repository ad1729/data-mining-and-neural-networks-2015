%% Santa Fe time series prediction

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