% -------------------------------------------------------------------------
% Homework2
% File : ExtendedKalmanFilterLocalization.m
%
% Discription : Mobible robot localization sample code with
% Extended Kalman Filter (EKF)
% -------------------------------------------------------------------------
 
function [] = ExtendedKalmanFilterLocalization()
 
close all;
clear all;

%设定仿真时间及步长
time = 0;
endtime = 60; % [sec]
global dt;
dt = 0.1; % [sec]
nSteps = ceil((endtime - time)/dt);

%结果初始化
result.time=[];
result.xTrue=[]; % 机器人真实位姿
result.xd=[];
result.xEst=[];  %机器人估计位姿
result.z=[];     %观测值
result.PEst=[];  %估计协方差
result.u=[];     %控制量
  
% State Vector [x y yaw v]'
xEst=[0 0 0 0]'; 
PEst = eye(4);
% True State
xTrue=xEst; 
% Dead Reckoning
xd=xTrue; 
% Observation vector [x y yaw v]'
z=[0 0 0 0]'; 
% Covariance Matrix for motion
Q=diag([0.1 0.1 toRadian(1) 0.05]).^2; 
% Covariance Matrix for observation
R=diag([1.5 1.5 toRadian(3) 0.05]).^2;
 
% Simulation parameter
global Qsigma
Qsigma=diag([0.1 toRadian(20)]).^2; %[v yawrate] 
global Rsigma
Rsigma=diag([1.5 1.5 toRadian(3) 0.05]).^2;%[x y z yaw v]
 
tic; %开始计时

% Main loop
for i=1 : nSteps
    time = time + dt;
    % 仿真控制输入Input
    u=doControl(time);
    % 仿真机器人运动及观察，获得Observation 
    [z,xTrue,xd,u]=Observation(xTrue, xd, u);
    
    %-------待补充如下代码---------------
    %-----------------------------------
    % -------- Kalman Filter -----------
    % step1：预测
    % （1）计算出当前时刻机器人位姿的预测值，xPred
    xPred = f(xEst, u);
    % （2）计算上述预测值的协方差，PPred
    jF = jacobF(xEst, u);
    PPred = jF * PEst * jF' + Q;

    % step2：观测
    % （1）计算真实观测和预测观测间的误差，即新息，y
    y = z - h(xPred);
    % （2）计算新息的协方差，S
    jH = jacobH(xPred);
    S = jH * PPred * jH' + R;

    % step3：更新
    % （1）更新机器人位姿预测为估计值，xEst
    K = PPred * jH' / S; % 计算卡尔曼增益
    xEst = xPred + K * y;
    % （2）更新估计值的协方差，PEst
    PEst = (eye(size(K,1)) - K * jH) * PPred;

  
    
    % Simulation Result
    result.time=[result.time; time];
    result.xTrue=[result.xTrue; xTrue'];
    result.xd=[result.xd; xd'];
    result.xEst=[result.xEst;xEst'];
    result.z=[result.z; z'];
    result.PEst=[result.PEst; diag(PEst)'];
    result.u=[result.u; u'];
    
    %Animation (remove some flames)
    if rem(i,5)==0
        plot(result.xTrue(:,1),result.xTrue(:,2),'.b');hold on;
        plot(result.z(:,1),result.z(:,2),'.g');hold on;
        plot(result.xd(:,1),result.xd(:,2),'.k');hold on;
        plot(result.xEst(:,1),result.xEst(:,2),'.r');hold on;
        ShowErrorEllipse(xEst,PEst);
        axis equal;
        grid on;
        drawnow;
    end 
end
toc %计时结束
%movie2avi(mov,'movie.avi'); %把仿真动画存为视频文件。 
DrawGraph(result);

function x = f(x, u)
% 理想运动模型Motion Model
global dt; 
F = [1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 0]; 
B = [
    dt*cos(x(3)) 0
    dt*sin(x(3)) 0
    0 dt
    1 0]; 
x= F*x+B*u;
 
function jF = jacobF(x, u)
% 运动模型的雅克比矩阵 Jacobian of Motion Model
global dt; 
jF=[
    1 0 0 0
    0 1 0 0
    -dt*u(1)*sin(x(3)) dt*u(1)*cos(x(3)) 1 0
     dt*cos(x(3)) dt*sin(x(3)) 0 1];

function z = h(x)
%理想观测模型Observation Model
H = [1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1 ]; 
z=H*x;

function jH = jacobH(x)
%观测模型的雅克比矩阵Jacobian of Observation Model
jH =[1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1];

%运动控制仿真，根据时间来仿真运动控制量u
function u = doControl(time)
%Calc Input Parameter
T=10; % [sec] 
% [V yawrate]
V=1.0; % [m/s]
yawrate = 5; % [deg/s] 
u =[ V*(1-exp(-time/T)) toRadian(yawrate)*(1-exp(-time/T))]';
 
%观测，真实的观测结果，带有噪声的。
function [z, x, xd, u] = Observation(x, xd, u)
%Calc Observation from noise prameter
global Qsigma;
global Rsigma;
x=f(x, u);% 没有噪声的情况，利用运动模型来计算机器人的位姿x，即真实值
u=u+Qsigma*randn(2,1);%add Process Noise
xd=f(xd, u);% 仿真Dead Reckoning，带有噪声，利用运动模型来预测机器人位姿
z=h(x+Rsigma*randn(4,1));%仿真观察Simulate Observation


function []=DrawGraph(result)
%Plot Result
figure(1);
x=[ result.xTrue(:,1:2) result.xEst(:,1:2) result.z(:,1:2)];
set(gca, 'fontsize', 16, 'fontname', 'times');
plot(x(:,5), x(:,6),'.g','linewidth', 4); hold on;
plot(x(:,1), x(:,2),'-.b','linewidth', 4); hold on;
plot(x(:,3), x(:,4),'r','linewidth', 4); hold on;
plot(result.xd(:,1), result.xd(:,2),'--k','linewidth', 4); hold on;
 
title('EKF Localization Result', 'fontsize', 16, 'fontname', 'times');
xlabel('X (m)', 'fontsize', 16, 'fontname', 'times');
ylabel('Y (m)', 'fontsize', 16, 'fontname', 'times');
legend('Ground Truth','GPS','Dead Reckoning','EKF','Error Ellipse');
grid on;
axis equal;

function angle=Pi2Pi(angle)
angle = mod(angle, 2*pi);
i = find(angle>pi);
angle(i) = angle(i) - 2*pi;
i = find(angle<-pi);
angle(i) = angle(i) + 2*pi;

function radian = toRadian(degree)
% degree to radian
radian = degree/180*pi;

function degree = toDegree(radian)
% radian to degree
degree = radian/pi*180;