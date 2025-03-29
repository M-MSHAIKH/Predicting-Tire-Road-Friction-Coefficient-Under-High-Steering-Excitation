% Creating the the model which takes input 4 files(Wet, Ice, Ice Sheet, Dry) and
% Predict they all surafces friction coefficinet
% dividing this into train test dataset
% training from 10sec, from where the actual experiement begin until 40

% Using only one value of mu. (a mean value of mu for each file or
% datasets). so the mu will be constnat through entire datasets.
%%%%%%%%%%%%%%%%%%%   LSTM   %%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

% Load all craig files

load_ice = load('hal_2018-12-12_ae.mat');
load_wet = load('hal_2018-12-12_ad.mat');
load_sheet = load('hal_2018-12-12_ai.mat');
load_dry = load('hal_2018-12-12_ac.mat');

%% Training set

% creating input datasets all files

% First Ice then wet then ICE cubes this ensures better optimization as ice
% and ice sheets gives you the small values, if both trained at the end
% then weights will not optimized well for dry surface.

% For consistency of the datalength we assume that we predict the friction
% until 40.500 sec or 20,251 time steps
% chnaging the position of the testing datasets
a=2;
b=3;
c=4;
d=1;
num_start = 5001;   %start from the 10 sec
num_train = floor((20251 - 5001) * 0.70);
num_valid = num_train + floor((20251 - 5001) * 0.10);   %start validation data point
num_steps = 20251;
left_steering_angles = [load_ice.y(num_start:num_steps,99),load_wet.y(num_start:num_steps,99),load_sheet.y(num_start:num_steps,99),load_dry.y(num_start:num_steps,99)];
L_angle_train = vertcat(left_steering_angles(1:num_train,1),left_steering_angles(1:num_train,2),left_steering_angles(1:num_train,3),left_steering_angles(1:num_train,4));
L_angle_valid = vertcat(left_steering_angles(num_train+1:num_valid,1),left_steering_angles(num_train+1:num_valid,2),left_steering_angles(num_train+1:num_valid,3),left_steering_angles(num_train+1:num_valid,4));
L_angle_test = vertcat(left_steering_angles(num_valid+1:end,a),left_steering_angles(num_valid+1:end,b),left_steering_angles(num_valid+1:end,c),left_steering_angles(num_valid+1:end,d));

right_steering_angle = [load_ice.y(num_start:num_steps,100),load_wet.y(num_start:num_steps,100),load_sheet.y(num_start:num_steps,100),load_dry.y(num_start:num_steps,100)];
R_angle_train = vertcat(right_steering_angle(1:num_train,1),right_steering_angle(1:num_train,2),right_steering_angle(1:num_train,3),right_steering_angle(1:num_train,4));
R_angle_valid = vertcat(right_steering_angle(num_train+1:num_valid,1),right_steering_angle(num_train+1:num_valid,2),right_steering_angle(num_train+1:num_valid,3),right_steering_angle(num_train+1:num_valid,4));
R_angle_test = vertcat(right_steering_angle(num_valid+1:end,a),right_steering_angle(num_valid+1:end,b),right_steering_angle(num_valid+1:end,c),right_steering_angle(num_valid+1:end,d));

left_steering_torques = [load_ice.y(num_start:num_steps,116),load_wet.y(num_start:num_steps,116),load_sheet.y(num_start:num_steps,116),load_dry.y(num_start:num_steps,116)];
L_torque_train = vertcat(left_steering_torques(1:num_train,1),left_steering_torques(1:num_train,2),left_steering_torques(1:num_train,3),left_steering_torques(1:num_train,4));
L_torque_valid = vertcat(left_steering_torques(num_train+1:num_valid,1),left_steering_torques(num_train+1:num_valid,2),left_steering_torques(num_train+1:num_valid,3),left_steering_torques(num_train+1:num_valid,4));
L_torque_test = vertcat(left_steering_torques(num_valid+1:end,a),left_steering_torques(num_valid+1:end,b),left_steering_torques(num_valid+1:end,c),left_steering_torques(num_valid+1:end,d));

right_steering_torques = [load_ice.y(num_start:num_steps,117),load_wet.y(num_start:num_steps,117),load_sheet.y(num_start:num_steps,117),load_dry.y(num_start:num_steps,117)];
R_torque_train = vertcat(right_steering_torques(1:num_train,1),right_steering_torques(1:num_train,2),right_steering_torques(1:num_train,3),right_steering_torques(1:num_train,4));
R_torque_valid = vertcat(right_steering_torques(num_train+1:num_valid,1),right_steering_torques(num_train+1:num_valid,2),right_steering_torques(num_train+1:num_valid,3),right_steering_torques(num_train+1:num_valid,4));
R_torque_test = vertcat(right_steering_torques(num_valid+1:end,a),right_steering_torques(num_valid+1:end,b),right_steering_torques(num_valid+1:end,c),right_steering_torques(num_valid+1:end,d));

time_sec = [load_ice.t(num_start:num_steps),load_wet.t(num_start:num_steps),load_sheet.t(num_start:num_steps),load_dry.t(num_start:num_steps)];
time_secIN = vertcat(load_ice.t(1:num_steps),load_wet.t(1:num_steps),load_sheet.t(1:num_steps),load_dry.t(1:num_steps));

gradient_time = 0.0020; 
delta_dot_iceL = gradient(left_steering_angles(:,1)) ./ gradient_time;


delta_dot_iceR = gradient(right_steering_angle(:,1)) ./ gradient_time;
delta_dot_wetL = gradient(left_steering_angles(:,2)) ./ gradient_time;
delta_dot_wetR = gradient(right_steering_angle(:,2)) ./ gradient_time;
delta_dot_sheetL = gradient(left_steering_angles(:,3)) ./ gradient_time;
delta_dot_sheetR = gradient(right_steering_angle(:,3)) ./ gradient_time;
delta_dot_dryL = gradient(left_steering_angles(:,4)) ./ gradient_time;
delta_dot_dryR = gradient(right_steering_angle(:,4)) ./ gradient_time;
left_ddot = [delta_dot_iceL,delta_dot_wetL,delta_dot_sheetL,delta_dot_dryL];
L_ddot_train = vertcat(left_ddot(1:num_train,1),left_ddot(1:num_train,2),left_ddot(1:num_train,3),left_ddot(1:num_train,4));
L_ddot_valid = vertcat(left_ddot(num_train+1:num_valid,1),left_ddot(num_train+1:num_valid,2),left_ddot(num_train+1:num_valid,3),left_ddot(num_train+1:num_valid,4));
L_ddot_test = vertcat(left_ddot(num_valid+1:end,a),left_ddot(num_valid+1:end,b),left_ddot(num_valid+1:end,c),left_ddot(num_valid+1:end,d));

right_ddot = [delta_dot_iceR,delta_dot_wetR,delta_dot_sheetR,delta_dot_dryR];
R_ddot_train = vertcat(right_ddot(1:num_train,1),right_ddot(1:num_train,2),right_ddot(1:num_train,3),right_ddot(1:num_train,4));
R_ddot_valid = vertcat(right_ddot(num_train+1:num_valid,1),right_ddot(num_train+1:num_valid,2),right_ddot(num_train+1:num_valid,3),right_ddot(num_train+1:num_valid,4));
R_ddot_test = vertcat(right_ddot(num_valid+1:end,a),right_ddot(num_valid+1:end,b),right_ddot(num_valid+1:end,c),right_ddot(num_valid+1:end,d));


% Calculating mu
mu_ice = 0.181976 * ones(num_steps-num_start,1);
mu_wet = 0.715519 * ones(num_steps-num_start,1);
mu_sheet = 0.246479 * ones(num_steps-num_start,1);
mu_dry = 0.504774 * ones(num_steps-num_start,1);
mu_train = vertcat(mu_ice(1:num_train,1),mu_wet(1:num_train,1),mu_sheet(1:num_train,1),mu_dry(1:num_train,1));
mu_valid = vertcat(mu_ice(num_train+1:num_valid,1),mu_wet(num_train+1:num_valid,1),mu_sheet(num_train+1:num_valid,1),mu_dry(num_train+1:num_valid,1));
% mu_test = vertcat(mu_ice(num_valid+1:end),mu_wet(num_valid+1:end),mu_sheet(num_valid+1:end),mu_dry(num_valid+1:end));
mu_test = vertcat(mu_wet(num_valid+1:end),0.181976,mu_sheet(num_valid+1:end),0.715519,mu_dry(num_valid+1:end),0.246479,mu_ice(num_valid+1:end),0.504774);

% 2341

% Input and output datset
validIN_data = [L_angle_valid,R_angle_valid, L_torque_valid, R_torque_valid, L_ddot_valid,R_ddot_valid];
validOUT_data = mu_valid;
testIN_data = [L_angle_test,R_angle_test, L_torque_test, R_torque_test, L_ddot_test,R_ddot_test];
testOUT_data = mu_test;
%% %% Normalization
norm_left_steering_angleIN = normalize(L_angle_train,'range',[-1,1]);
min_langle = min(L_angle_train);
max_langle = max(L_angle_train);
norm_right_steering_angleIN = normalize(R_angle_train,'range',[-1,1]);
min_rangle = min(R_angle_train);
max_rangle = max(R_angle_train);
norm_left_ddotIN = normalize(L_ddot_train,'range',[-1,1]);
min_lddot = min(L_ddot_train);
max_lddot = max(L_ddot_train);
norm_right_ddotIN = normalize(R_ddot_train,'range',[-1,1]);
min_rddot = min(R_ddot_train);
max_rddot = max(R_ddot_train);
norm_left_steering_torquesIN = normalize(L_torque_train,'range',[-1,1]);
min_ltorques = min(L_torque_train);
max_ltorques = max(L_torque_train);
norm_right_steering_torquesIn = normalize(R_torque_train,'range',[-1,1]);
min_rtorques = min(R_torque_train);
max_rtorques = max(R_torque_train);
norm_mu = normalize(mu_train,'range',[-1,1]);
min_mu = min(mu_train);
max_mu = max(mu_train);

% Input and output training datasets
input_data_train = [norm_left_steering_angleIN,norm_right_steering_angleIN, norm_left_ddotIN, norm_right_ddotIN, norm_left_steering_torquesIN,norm_right_steering_torquesIn];
% input_data_train = input_data_train';
output_data_train = norm_mu;
% output_data_train = output_data_train';

%% Normalizing the validation and the test data
% use -4 for test data output

% Normlaizing the validation data
norm_left_steering_angle_valid = zeros(length(L_angle_valid),1);
norm_right_steering_angle_valid = zeros(length(L_angle_valid),1);
norm_left_ddot_valid = zeros(length(L_angle_valid),1);
norm_right_ddot_valid = zeros(length(L_angle_valid),1);
norm_left_steering_torques_valid = zeros(length(L_angle_valid),1);
norm_right_steering_torques_valid = zeros(length(L_angle_valid),1);

for i = 1:length(L_angle_valid)
    norm_left_steering_angle_valid(i) = rangeRescale(L_angle_valid(i),min_langle,max_langle,-1,1);
    norm_right_steering_angle_valid(i) = rangeRescale(R_angle_valid(i),min_rangle,max_rangle,-1,1);
    norm_left_ddot_valid(i) = rangeRescale(L_ddot_valid(i),min_lddot,max_lddot,-1,1);
    norm_right_ddot_valid(i) = rangeRescale(R_ddot_valid(i),min_rddot,max_rddot,-1,1);
    norm_left_steering_torques_valid(i) = rangeRescale(L_torque_valid(i),min_ltorques,max_ltorques,-1,1);
    norm_right_steering_torques_valid(i) = rangeRescale(R_torque_valid(i),min_rtorques,max_rtorques,-1,1);
end

input_valid = [norm_left_steering_angle_valid,norm_right_steering_angle_valid, norm_left_ddot_valid,norm_right_ddot_valid, norm_left_steering_torques_valid, norm_right_steering_torques_valid];
% input_valid = input_valid';
output_valid = mu_valid;

% Normlaizing the test data
norm_left_steering_angle_test = zeros(length(L_angle_test),1);
norm_right_steering_angle_test = zeros(length(L_angle_test),1);
norm_left_ddot_test = zeros(length(L_angle_test),1);
norm_right_ddot_test = zeros(length(L_angle_test),1);
norm_left_steering_torques_test = zeros(length(L_angle_test),1);
norm_right_steering_torques_test = zeros(length(L_angle_test),1);

for i = 1:length(L_angle_test)
    norm_left_steering_angle_test(i) = rangeRescale(L_angle_test(i),min_langle,max_langle,-1,1);
    norm_right_steering_angle_test(i) = rangeRescale(R_angle_test(i),min_rangle,max_rangle,-1,1);
    norm_left_ddot_test(i) = rangeRescale(L_ddot_test(i),min_lddot,max_lddot,-1,1);
    norm_right_ddot_test(i) = rangeRescale(R_ddot_test(i),min_rddot,max_rddot,-1,1);
    norm_left_steering_torques_test(i) = rangeRescale(L_torque_test(i),min_ltorques,max_ltorques,-1,1);
    norm_right_steering_torques_test(i) = rangeRescale(R_torque_test(i),min_rtorques,max_rtorques,-1,1);
end

input_test = [norm_left_steering_angle_test,norm_right_steering_angle_test, norm_left_ddot_test,norm_right_ddot_test, norm_left_steering_torques_test, norm_right_steering_torques_test];
% input_test = input_test';
output_test = mu_test;
%% LSTM Architecture

numResponses = 1;
featureDimension = 6;
numHiddenUnits = 100;
maxEpochs = 500;
miniBatchSize = 200;
% sigmoidLayer
Networklayers = [
    sequenceInputLayer(featureDimension)
    lstmLayer(100)
    fullyConnectedLayer(1)
];
%L2 Regularization (also known as Weight Decay) is a technique used to reduce overfitting by adding a penalty term to the loss function that discourages large weights. This helps prevent the model from fitting too closely to the training data and improves generalization to unseen data.

options = trainingOptions('adam', ...
    'L2Regularization',1.0000e-04,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',10, ...
    'Shuffle','never', ...
    'Plots','training-progress',...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',100,...
    'Verbose',0,...
    'ValidationData', [{input_valid} {output_valid}],...
    'ValidationFrequency',30);

%Training a LSTM
% lstm_net = trainnet(input_data_train,output_data_train,Networklayers,'mse',options);

%% Predictions

% Loading the previously trained LSTM
load_lstm = load("lstm_28_03.mat");
lstm_net = load_lstm.lstm_net;


% view(net_train);
y_norm = predict(lstm_net,input_test);
% y_norm = y_norm';

% Renormalized to its standard value using the training min max
y_output = zeros(length(y_norm),1);
for s = 1:length(y_norm)
    y_output(s,1) = (((1 + y_norm(s,1)) / 2) * (max_mu - min_mu)) + min_mu;
end
%% figure;
figure;
plot(y_output,'.');
hold on;
plot(output_test,'.')
title('Mu estimation Using LSTM')
% xlabel('Time(sec)')
ylabel('mu Estimated')
% xlim([10,40])
legend('NN Output of mu', 'True Values of mu')
grid on;


%% %% WEt,shet,dry,ice
% Time vector for the plot
time_plot1 = time_sec(num_valid+1:end,1);
num_val_indi = num_valid/4 + 1;

h1 = figure;
hold on;
plot(time_plot1,y_output(1:num_val_indi),'.','Color','b');
plot(time_plot1,y_output(num_val_indi+1 : num_val_indi*2),'.','Color','g');
plot(time_plot1,y_output(num_val_indi*2+1 : num_val_indi*3),'.','Color','r');
plot(time_plot1,y_output(num_val_indi*3+1 : end),'.','Color','m');
plot(time_plot1,output_test(1:num_val_indi),'*','Color','b');
plot(time_plot1,output_test(num_val_indi+1 : num_val_indi*2),'*','Color','g');
plot(time_plot1,output_test(num_val_indi*2+1 : num_val_indi*3),'*','Color','r');
plot(time_plot1,output_test(num_val_indi*3+1 : end),'*','Color','m');
title('\mu estimation using LSTM')
% xlabel('Time(sec)')
ylabel('\mu estimated')
xlabel('Time (sec)')
% xlim([10,40])
legend('LSTM prediction for wet surface','LSTM prediction for ice sheet surface',...
    'LSTM prediction for dry surface','LSTM prediction for ice surface',...
    'True values for wet surface', 'True values for ice sheet surface',...
    'True values for dry surface','True values for ice surface')
ylim([0 1.2])
grid on;
% exportgraphics(h1,'lstm_predictions.png','Resolution',300)




