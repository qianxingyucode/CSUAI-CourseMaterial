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

%�趨����ʱ�估����
time = 0;
endtime = 60; % [sec]
global dt;
dt = 0.1; % [sec]
nSteps = ceil((endtime - time)/dt);

%�����ʼ��
result.time=[];
result.xTrue=[]; % ��������ʵλ��
result.xd=[];
result.xEst=[];  %�����˹���λ��
result.z=[];     %�۲�ֵ
result.PEst=[];  %����Э����
result.u=[];     %������
  
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
 
tic; %��ʼ��ʱ

% Main loop
for i=1 : nSteps
    time = time + dt;
    % �����������Input
    u=doControl(time);
    % ����������˶����۲죬���Observation 
    [z,xTrue,xd,u]=Observation(xTrue, xd, u);
    
    %-------���������´���---------------
    %-----------------------------------
    % -------- Kalman Filter -----------
    % step1��Ԥ��
    % ��1���������ǰʱ�̻�����λ�˵�Ԥ��ֵ��xPred
    xPred = f(xEst, u);
    % ��2����������Ԥ��ֵ��Э���PPred
    jF = jacobF(xEst, u);
    PPred = jF * PEst * jF' + Q;

    % step2���۲�
    % ��1��������ʵ�۲��Ԥ��۲���������Ϣ��y
    y = z - h(xPred);
    % ��2��������Ϣ��Э���S
    jH = jacobH(xPred);
    S = jH * PPred * jH' + R;

    % step3������
    % ��1�����»�����λ��Ԥ��Ϊ����ֵ��xEst
    K = PPred * jH' / S; % ���㿨��������
    xEst = xPred + K * y;
    % ��2�����¹���ֵ��Э���PEst
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
toc %��ʱ����
%movie2avi(mov,'movie.avi'); %�ѷ��涯����Ϊ��Ƶ�ļ��� 
DrawGraph(result);

function x = f(x, u)
% �����˶�ģ��Motion Model
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
% �˶�ģ�͵��ſ˱Ⱦ��� Jacobian of Motion Model
global dt; 
jF=[
    1 0 0 0
    0 1 0 0
    -dt*u(1)*sin(x(3)) dt*u(1)*cos(x(3)) 1 0
     dt*cos(x(3)) dt*sin(x(3)) 0 1];

function z = h(x)
%����۲�ģ��Observation Model
H = [1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1 ]; 
z=H*x;

function jH = jacobH(x)
%�۲�ģ�͵��ſ˱Ⱦ���Jacobian of Observation Model
jH =[1 0 0 0
    0 1 0 0
    0 0 1 0
    0 0 0 1];

%�˶����Ʒ��棬����ʱ���������˶�������u
function u = doControl(time)
%Calc Input Parameter
T=10; % [sec] 
% [V yawrate]
V=1.0; % [m/s]
yawrate = 5; % [deg/s] 
u =[ V*(1-exp(-time/T)) toRadian(yawrate)*(1-exp(-time/T))]';
 
%�۲⣬��ʵ�Ĺ۲��������������ġ�
function [z, x, xd, u] = Observation(x, xd, u)
%Calc Observation from noise prameter
global Qsigma;
global Rsigma;
x=f(x, u);% û������������������˶�ģ������������˵�λ��x������ʵֵ
u=u+Qsigma*randn(2,1);%add Process Noise
xd=f(xd, u);% ����Dead Reckoning�����������������˶�ģ����Ԥ�������λ��
z=h(x+Rsigma*randn(4,1));%����۲�Simulate Observation


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