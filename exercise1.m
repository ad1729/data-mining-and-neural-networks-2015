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
title('y = sin(0.7\pix) with x in [0,1]');
print('\home\ad\Desktop\images\sinpix', '-dpng');

%% training network with 1 hidden layer and 2 neurons
net = fitnet(2);
net = configure(net, x, y);
net.inputs{1}.processFcns = {};
net.outputs{2}.processFcns = {};
[net, tr] = train(net, x, y);

%% layer weights and biases; activation functions
% hidden layer with two neurons
[biases, weights] = hidden_layer_weights(net);
transfer_function = hidden_layer_transfer_function(net);
str2func(net.layers{1}.transferFcn);
str2func(net.layers{2}.transferFcn);

% output layer with single node
[biases_out, weights_out] = output_layer_weights(net);
transfer_out = output_layer_transfer_function(net);
str2func(net.layers{end}.transferFcn);

%% Calculations using the previous section

x1 = x * weights(1) + biases(1);
x2 = x * weights(2) + biases(2);

x_p = transfer_function(x1 + x2);
output_p = transfer_out(x_p * weights_out(1) + x_p * weights_out(2) + biases_out);

plot(x, output_p, 'r-');
title('Plot of output_p vs input vector x');
ylabel('Output_p');
xlabel('x');
print('\home\ad\Desktop\images\outputPvsX', '-dpng');

%% (3) Function Approximation (noisy case)
%% Set parameter options; reate a training and validation set
data_size = 1000;
std_dev = 0.5;
train_algorithm = 'trainscg'; % trainscg, trainlm, trainrp, trainbfg
regularization_parameter = 0.000001; % in [0,1]
neurons = 5;

train_x = linspace(-1, 1, data_size);
train_y = sin(2 * pi * train_x) + (std_dev * randn(size(train_x)));
val_x = linspace(-0.9, 0.9, data_size);
val_y = sin(2 * pi * val_x) + (std_dev * randn(size(val_x)));
noise_x = [train_x val_x];
noise_y = [train_y val_y];

%% fit the model
new_net = fitnet(neurons, train_algorithm); 
new_net.divideFcn = 'divideind';
%new_net.divideParam = struct('trainInd', 1:100, ...
%    'valInd', 101:200, ...
%    'testInd', []); % no test set
new_net.divideParam = struct('trainInd', 1:data_size, ...
    'valInd', (data_size + 1):(data_size * 2), ...
    'testInd', []); % no test set
new_net.performParam.regularization = regularization_parameter;
% Cost function with regularization. net.performParam.regularization can take on
% any value between 0 and 1 inclusive. The default value of 0 corresponds to no regularization, 1
% will yield a network with all weights and biases set to zero.

[new_net, new_tr] = train(new_net, noise_x, noise_y);

%% Get approximated function on training set
title_string = strcat('nodes=', num2str(neurons), {', '}, '\sigma=', num2str(std_dev), ...
    {', '}, 'nobs=', num2str(data_size), {', '}, '\lambda=', num2str(regularization_parameter), ...
    {', '}, 'alg=', train_algorithm);
train_y_hat = new_net(train_x);
plot(train_x, train_y, 'r-');
hold on;
plot(train_x, train_y_hat, '-');
plot(train_x, sin(2 * pi * train_x), 'g-') ;
hold off;
title(title_string);
legend('Training Set', 'Approximated Function', 'True Function');


