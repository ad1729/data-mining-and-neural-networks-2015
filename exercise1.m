%% (1.1) Function Approximation (noiseless case)
clear;
clc;
nnd11gn

%% (1.2) Role of the hidden layer and the output layer
clear;
clc;

x = linspace(0,1,21);
y = sin(0.7 * pi * x);
plot(x, y, 'b-*');
title('y = sin(0.7\pix) with x \in [0,1]');
xlabel('x');
ylabel('y');
%print('\home\ad\Desktop\images\sinpix', '-dpng');

%% training network with 1 hidden layer and 2 neurons
net = fitnet(2);
net = configure(net, x, y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, ~] = train(net, x, y);

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
x1 = x * weights(1) + biases(1);
x2 = x * weights(2) + biases(2);

x1p = transfer_function(x1);
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


