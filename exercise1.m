%% (1.1) Function Approximation (noiseless case)
clear;
clc;
nnd11gn

%% (1.2) Role of the hidden layer and the output layer
clear;
clc;

x = linspace(0,1,21);
y1 = sin(0.7 * pi * x);
plot(x, y1, 'b-*');
title('y = sin(0.7\pix) with x \in [0,1]');
xlabel('x');
ylabel('y');
%print('\home\ad\Desktop\images\sinpix', '-dpng');

%% training network with 1 hidden layer and 2 neurons
net = fitnet(2);
net = configure(net, x, y1);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, ~] = train(net, x, y1);

%% layer weights and biases; activation functions
% hidden layer with two neurons
[biases, weights] = hidden_layer_weights(net);
transfer_function = hidden_layer_transfer_function(net);
str2func(net.layers{1}.transferFcn) % hidden layer

% output layer with single node
[biases_out, weights_out] = output_layer_weights(net);
transfer_out = output_layer_transfer_function(net);
str2func(net.layers{end}.transferFcn) % output layer

%% Calculations using the previous section
p = 1:21;
x1_train = x * weights(1) + biases(1);
x2 = x * weights(2) + biases(2);

x1p = transfer_function(x1_train);
plot(p, x1p, '*');
xlabel('p');
ylabel('x_p^1');
%print('\home\ad\Desktop\images\x1p', '-dpng');

x2p = transfer_function(x2);
plot(p, x2p, '*');
xlabel('p');
ylabel('x_p^2');
%print('\home\ad\Desktop\images\x2p', '-dpng');

plot(p, x1p + x2p, '*');
xlabel('p');
ylabel('y_p');
%print('\home\ad\Desktop\images\yp', '-dpng');

output_p = transfer_out(weights_out(1) * x1p + weights_out(2) * x2p + biases_out);
plot(x, output_p, 'r*-');
title('Plot of output_p vs input vector x');
ylabel('output_p');
xlabel('x');
%print('\home\ad\Desktop\images\outputp', '-dpng');

%% (1.3) Function Approximation (noisy case)
clear;
clc;

%% Fixing the initial seed for the rng
rng(1,'twister');
s = rng;

%% Set parameter options; create a training and validation set
data_size = 100000; % 50, 200, 1000, 100000 (final part for comparing algorithms)
std_dev = 1.0; % 0.1, 1.0, 2.0
train_algorithm = 'trainbfg'; % trainscg, trainlm, trainbfg trainrp
regularization_parameter = 1e-3; % in [0,1], 0.8, 0.1, 1e-4
neurons = 40; % 5, 10, 20, 40

%% Create dataset
rng(s); % cuts down on the randn variability
train_x = linspace(-1, 1, data_size);
train_y = sin(2 * pi * train_x) + (std_dev * randn(size(train_x)));
val_x = linspace(-0.9, 0.9, data_size);
val_y = sin(2 * pi * val_x) + (std_dev * randn(size(val_x)));
noise_x = [train_x val_x];
noise_y = [train_y val_y];

%% fit the model
net = fitnet(neurons, train_algorithm); 

% early stopping or regularization
net.divideFcn = 'dividetrain'; % 'dividetrain' for comparing the algorithm runtimes, 'divideind' for others
net.divideParam = struct('trainInd', 1:data_size, ...
    'valInd', (data_size + 1):(data_size * 2), ...
    'testInd', []); % no test set
%net.performParam.regularization = regularization_parameter;
net.trainParam.epochs = 50; % 50, 200 for assessing training times

%training the network
[net, tr] = train(net, noise_x, noise_y);

%% Get approximated function on training set
title_string = strcat('nodes=', num2str(neurons), {', '}, '\sigma=', num2str(std_dev), ...
    {', '}, 'nobs=', num2str(data_size), {', '}, '\lambda=', num2str(regularization_parameter), ...
    {', '}, 'alg=', train_algorithm);
train_y_hat = net(train_x);
plot(train_x, train_y, '*');
hold on;
plot(train_x, train_y_hat, 'r+-', 'LineWidth', 0.5);
plot(train_x, sin(2 * pi * train_x), 'g*-', 'LineWidth', 1);
hold off;
title(title_string);
legend('Training Set', 'Approximated Function', 'True Function');
xlabel('x');
ylabel('sin(2\pix) + noise');
%print('\home\ad\Desktop\images\o', '-dpng');

%% (1.4) Curse of dimensionality
clear;
clc;

%% m = 1
x1_train = linspace(-5,5,100);
x1_test = linspace(-4.9,4.9,53);

intersect(x1_train, x1_test); % test if there is overlap between the training and test sets

y1_train = sinc(x1_train);
y1_test = sinc(x1_test); 

x1 = [x1_train, x1_test];
y1 = [y1_train, y1_test];

plot(x1_train,y1_train, '*-', 'MarkerSize', 6);
hold on;
plot(x1_test,y1_test, 'r+-', 'MarkerSize', 10);
hold off;
xlabel('x \in [-5,5]');
ylabel('sinc(x)');
title('Sinc Function, m = 1')
legend('Training set', 'Test set');
print('\home\ad\Desktop\images\sinc1', '-dpng');

%% fit net 1
net = fitnet(5); % 5
net.layers{1}.transferFcn ='radbas';
net.divideFcn = 'dividetrain';
% net.divideParam = struct('trainInd', 1:100, ...
%     'valInd', [], ...
%     'testInd', []); % no validation set
%net.performParam.regularization = 1e-6;

[net, tr] = train(net, x1_train, y1_train);
%perf = perform(net, x1, y1);
%perf_test = perform(net, x1_test, y1_test);

% approximated function
plot(x1_train,y1_train, '-'); % training
hold on;
plot(x1_train, net(x1_train), '-r', 'LineWidth', 1);
plot(x1_test, net(x1_test), '--g', 'LineWidth', 2);
hold off;
legend('Training Set', 'Fitted Func.', 'Test Set Fit');
xlabel('x');
ylabel('y = sinc(r)');
%print('\home\ad\Desktop\images\sinc1log', '-dpng');
%print('\home\ad\Desktop\images\sinc1radbas', '-dpng');

%% Curse of Dimensionality (contd.)
%% m = 2
[X1,X2] = meshgrid(linspace(-5,5,100));
R = sqrt(X1.^2 + X2.^2);
Z = sinc(R);

figure
mesh(X1, X2, Z)
%surf(X1, X2, Z)
xlabel('x1 \in [-5,5]');
ylabel('x2 \in [-5,5]');
zlabel('z = sinc(r)');
title('Sinc Function, m = 2');
%print('\home\ad\Desktop\images\sinc2', '-dpng');

%% Training, validation and test sets
train_x = [X1(:), X2(:)].';
temp1 = train_x.*train_x;
train_z = sinc(sqrt((temp1(1,:) + temp1(2,:))));

[val1,val2]=meshgrid(linspace(-4.9,4.9,53));
val_x = [val1(:), val2(:)].';
temp2 = val_x.*val_x;
val_z = sinc(sqrt((temp2(1,:) + temp2(2,:))));

% for training the network
inputs = [train_x val_x];
target = [train_z val_z];

[test1, test2] = meshgrid(linspace(-4.8,4.8,40));
test_x = [test1(:), test2(:)].';
temp3 = test_x.*test_x;
test_z = sinc(sqrt((temp3(1,:)+temp3(2,:))));

%% fit net 2
net = fitnet(25, 'trainlm'); % 25, 30, 35
net.layers{1}.transferFcn ='radbas';
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', 1:10000, ...
    'valInd', 10001:12809, ...
    'testInd', []); % no test set at the time of training
%net.performParam.regularization = 1e-2;
net.trainParam.epochs = 4000;

[net, tr] = train(net, inputs, target, 'useParallel', 'yes');

%% Mesh plot
Y_hat = net(test_x);

plot3(test_x(1,:),test_x(2,:),test_z,'b*', 'MarkerSize',1);
title('Mexican Hat Function')
xlabel('x1 \in [-4.8,4.8]')
ylabel('x2 \in [-4.8,4.8]')
zlabel('z = sinc(r)')
hold on;
plot3(test_x(1,:),test_x(2,:),Y_hat,'r--');
legend({'Test Set','Approx. Function'});
hold off;
%print('\home\ad\Desktop\images\sinc2fit35', '-dpng');

%% Plot by dimension
Y_hat = net(test_x);

figure;
subplot(1,2,1)
plot(test_x(1,:), test_z, 'o', test_x(1,:), Y_hat, '*');
xlabel('x1');
ylabel('Z');
title('Dimension 1');
legend('Test set', 'Approximated', 'Location', 'southeast');
subplot(1,2,2)
plot(test_x(2,:), test_z, 'o', test_x(2,:), Y_hat, '*');
xlabel('x2');
ylabel('Z');
title('Dimension 2');
legend('Test set', 'Approximated', 'Location', 'southeast');
%print('\home\ad\Desktop\images\sinc2point25', '-dpng');
display(tr.best_perf)
perf_test = perform(net, test_z, Y_hat)

%% m = 5
[X1,X2,X3,X4,X5] = ndgrid(linspace(-5,5,15));
train_x = [X1(:), X2(:), X3(:), X4(:), X5(:)].';
temp1 = train_x.*train_x;
train_z = sinc(sqrt((temp1(1,:) + temp1(2,:) + temp1(3,:) + temp1(4,:) + temp1(5,:))));

[val1,val2, val3, val4, val5] = ndgrid(linspace(-4.9,4.9,10));
val_x = [val1(:), val2(:), val3(:), val4(:), val5(:)].';
temp2 = val_x.*val_x;
val_z = sinc(sqrt((temp2(1,:) + temp2(2,:) + temp2(3,:) + temp2(4,:) + temp2(5,:))));

% for training the network
inputs = [train_x val_x];
target = [train_z val_z];

[test1, test2, test3, test4, test5] = ndgrid(linspace(-4.8,4.8,10));
test_x = [test1(:), test2(:), test3(:), test4(:), test5(:)].';
temp3 = test_x.*test_x;
test_z = sinc(sqrt((temp3(1,:) + temp3(2,:) + temp3(3,:) + temp3(4,:) + temp3(5,:))));

%% fit net 2
net = fitnet(300, 'trainscg'); % 100, 200, 300, 400, 500
net.layers{1}.transferFcn ='radbas';
net.divideFcn = 'divideind';
net.divideParam = struct('trainInd', 1:(15^5), ...
    'valInd', (15^5)+1:length(inputs), ...
    'testInd', []); % no test set at the time of training
%net.performParam.regularization = 1e-2;
net.trainParam.epochs = 500;

[net, tr] = train(net, inputs, target, 'useParallel', 'yes');

Y_hat = net(test_x);
format long
display(tr.best_perf)
perf_test = perform(net, test_z, Y_hat)