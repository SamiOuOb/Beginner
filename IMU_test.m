clc
clear all
close all

accel_record_UE1 = 'accel_record_UE1.txt';
accel_record_UE2 = 'accel_record_UE2.txt';
accel_record_UE3 = 'accel_record_UE3.txt';
accel_record_UE4 = 'accel_record_UE4.txt';
gyro_record_UE1 = 'gyro_record_UE1.txt';
gyro_record_UE2 = 'gyro_record_UE2.txt';
gyro_record_UE3 = 'gyro_record_UE3.txt';
gyro_record_UE4 = 'gyro_record_UE4.txt';
mag_record_UE1 = 'mag_record_UE1.txt';
mag_record_UE2 = 'mag_record_UE2.txt';
mag_record_UE3 = 'mag_record_UE3.txt';
mag_record_UE4 = 'mag_record_UE4.txt';
mic_record_UE1 = 'mic_record_UE1.txt';
mic_record_UE2 = 'mic_record_UE2.txt';
mic_record_UE3 = 'mic_record_UE3.txt';
mic_record_UE4 = 'mic_record_UE4.txt';
fid_accel_record_UE1 = fopen(accel_record_UE1,'r');
fid_accel_record_UE2 = fopen(accel_record_UE2,'r');
fid_accel_record_UE3 = fopen(accel_record_UE3,'r');
fid_accel_record_UE4 = fopen(accel_record_UE4,'r');
fid_gyro_record_UE1 = fopen(gyro_record_UE1,'r');
fid_gyro_record_UE2 = fopen(gyro_record_UE2,'r');
fid_gyro_record_UE3 = fopen(gyro_record_UE3,'r');
fid_gyro_record_UE4 = fopen(gyro_record_UE4,'r');
fid_mag_record_UE1 = fopen(mag_record_UE1,'r');
fid_mag_record_UE2 = fopen(mag_record_UE2,'r');
fid_mag_record_UE3 = fopen(mag_record_UE3,'r');
fid_mag_record_UE4 = fopen(mag_record_UE4,'r');
fid_mic_record_UE1 = fopen(mic_record_UE1,'r');
fid_mic_record_UE2 = fopen(mic_record_UE2,'r');
fid_mic_record_UE3 = fopen(mic_record_UE3,'r');
fid_mic_record_UE4 = fopen(mic_record_UE4,'r');

%%% Ploting Parameters %%%
UE1_color = [32, 107, 84] / 255  ;
UE2_color = [98, 171, 193] / 255 ;
UE3_color = [255, 0, 0] / 255 ; 
UE4_color = [254, 127, 45] / 255 ;

line_width = 1.2;
%%%%%%%%%%%%%%%%%%%%%%%%

%%% Acceleration Records %%%
AccelUE1 = readRecord(fid_accel_record_UE1);
AccelUE2 = readRecord(fid_accel_record_UE2);
AccelUE3 = readRecord(fid_accel_record_UE3);
AccelUE4 = readRecord(fid_accel_record_UE4);


AccelUE1_temp = ArrangeDataAndTime(AccelUE1);
AccelUE2_temp = ArrangeDataAndTime(AccelUE2);
AccelUE3_temp = ArrangeDataAndTime(AccelUE3);
AccelUE4_temp = ArrangeDataAndTime(AccelUE4);

Fs = 5;
t_interval = 1 / Fs;
%%%%%
newT1 = AccelUE1_temp(:,1);
newT2 = AccelUE2_temp(:,1);
newT3 = AccelUE3_temp(:,1);
newT4 = AccelUE4_temp(:,1);

T = unique([newT1' newT2' newT3' newT4']);
newT = 0:t_interval:T(end);

newAccelUE1 = dataInterpolation(newT,newT1,AccelUE1_temp);
newAccelUE2 = dataInterpolation(newT,newT2,AccelUE2_temp);
newAccelUE3 = dataInterpolation(newT,newT3,AccelUE3_temp);
newAccelUE4 = dataInterpolation(newT,newT4,AccelUE4_temp);

AccelUE1_sample = sampleByTime(newAccelUE1,t_interval);
AccelUE2_sample = sampleByTime(newAccelUE2,t_interval);
AccelUE3_sample = sampleByTime(newAccelUE3,t_interval);
AccelUE4_sample = sampleByTime(newAccelUE4,t_interval);

%%% time domain %%%
% figure(1)
% subplot(221)
% plot(AccelUE1_sample(:,1),AccelUE1_sample(:,2),AccelUE2_sample(:,1),AccelUE2_sample(:,2))
% legend('UE1','UE2')
% xlabel('Time (s)')
% ylabel('Acceleration (m/s^2)')
% subplot(222)
% plot(AccelUE3_sample(:,1),AccelUE3_sample(:,2),AccelUE4_sample(:,1),AccelUE4_sample(:,2))
% legend('UE3','UE4')
% xlabel('Time (s)')
% ylabel('Acceleration (m/s^2)')
% subplot(223)
% plot(AccelUE1_sample(:,1),AccelUE1_sample(:,2),AccelUE3_sample(:,1),AccelUE3_sample(:,2))
% legend('UE1','UE3')
% xlabel('Time (s)')
% ylabel('Acceleration (m/s^2)')
% subplot(224)
% plot(AccelUE2_sample(:,1),AccelUE2_sample(:,2),AccelUE3_sample(:,1),AccelUE3_sample(:,2))
% legend('UE2','UE3')
% xlabel('Time (s)')
% ylabel('Acceleration (m/s^2)')

AccelUE1_sum = sqrt(AccelUE1_sample(:,2).^2 + AccelUE1_sample(:,3).^2 +AccelUE1_sample(:,4).^2 );
AccelUE2_sum = sqrt(AccelUE2_sample(:,2).^2 + AccelUE2_sample(:,3).^2 +AccelUE2_sample(:,4).^2 );
AccelUE3_sum = sqrt(AccelUE3_sample(:,2).^2 + AccelUE3_sample(:,3).^2 +AccelUE3_sample(:,4).^2 );
AccelUE4_sum = sqrt(AccelUE4_sample(:,2).^2 + AccelUE4_sample(:,3).^2 +AccelUE4_sample(:,4).^2 );

AccelUE1_net = diff(AccelUE1_sum);
AccelUE2_net = diff(AccelUE2_sum);
AccelUE3_net = diff(AccelUE3_sum);
AccelUE4_net = diff(AccelUE4_sum);

% S1 = 1 - pdist([AccelUE1_sum , AccelUE2_sum]','cosine');
% S2 = 1 - pdist([AccelUE3_sum , AccelUE4_sum]','cosine');
% S3 = 1 - pdist([AccelUE1_sum , AccelUE3_sum]','cosine');
% S4 = 1 - pdist([AccelUE2_sum , AccelUE3_sum]','cosine');
% S = [S1 S2 S3 S4]

%%% PSD %%%
n = 4; % x:2, y:3, z:4
Accel_R1_temp = xcorr(AccelUE1_sample(:,n));
Accel_R2_temp = xcorr(AccelUE2_sample(:,n));
Accel_R3_temp = xcorr(AccelUE3_sample(:,n));
Accel_R4_temp = xcorr(AccelUE4_sample(:,n));
% 
AccelUE1_AutoCor = xcorr(AccelUE1_sum);
AccelUE2_AutoCor = xcorr(AccelUE2_sum);
AccelUE3_AutoCor = xcorr(AccelUE3_sum);
AccelUE4_AutoCor = xcorr(AccelUE4_sum);


x = (1 + length(Accel_R1_temp))/2;

Accel_R1 = Accel_R1_temp(1:x) ./ (1:x);
Accel_R2 = Accel_R2_temp(1:x) ./ (1:x);
Accel_R3 = Accel_R3_temp(1:x) ./ (1:x);
Accel_R4 = Accel_R4_temp(1:x) ./ (1:x);
% 
AccelUE1_Cor = AccelUE1_AutoCor(1:x) ./ (1:x);
AccelUE2_Cor = AccelUE2_AutoCor(1:x) ./ (1:x);
AccelUE3_Cor = AccelUE3_AutoCor(1:x) ./ (1:x);
AccelUE4_Cor = AccelUE4_AutoCor(1:x) ./ (1:x);
% 
% 
N = length(Accel_R1);
% 
% Accel_FFT1 = fft(Accel_R1);
% Accel_FFT1 = Accel_FFT1(1:N/2+1);
% Accel_FFT2 = fft(Accel_R2);
% Accel_FFT2 = Accel_FFT2(1:N/2+1);
% Accel_FFT3 = fft(Accel_R3);
% Accel_FFT3 = Accel_FFT3(1:N/2+1);
% Accel_FFT4 = fft(Accel_R4);
% Accel_FFT4 = Accel_FFT4(1:N/2+1);

Accel_FFT1 = fft(AccelUE1_Cor);
Accel_FFT1 = Accel_FFT1(1:N/2+1);
Accel_FFT2 = fft(AccelUE2_Cor);
Accel_FFT2 = Accel_FFT2(1:N/2+1);
Accel_FFT3 = fft(AccelUE3_Cor);
Accel_FFT3 = Accel_FFT3(1:N/2+1);
Accel_FFT4 = fft(AccelUE4_Cor);
Accel_FFT4 = Accel_FFT4(1:N/2+1);


Accel_PSD1 = (1/(Fs*N)) * abs(Accel_FFT1);
Accel_PSD1(2:end-1) = 2 * Accel_PSD1(2:end-1) ;
Accel_PSD1_dB = 10*log10(Accel_PSD1);
Accel_PSD2 = (1/(Fs*N)) * abs(Accel_FFT2);
Accel_PSD2(2:end-1) = 2 * Accel_PSD2(2:end-1) ;
Accel_PSD2_dB = 10*log10(Accel_PSD2);
Accel_PSD3 = (1/(Fs*N)) * abs(Accel_FFT3);
Accel_PSD3(2:end-1) = 2 * Accel_PSD3(2:end-1) ;
Accel_PSD3_dB = 10*log10(Accel_PSD3);
Accel_PSD4 = (1/(Fs*N)) * abs(Accel_FFT4);
Accel_PSD4(2:end-1) = 2 * Accel_PSD4(2:end-1) ;
Accel_PSD4_dB = 10*log10(Accel_PSD4);

AreaUE1 = 0.5 * [mean(Accel_PSD1_dB(1:63)) mean(Accel_PSD1_dB(64:126)) ...
                 mean(Accel_PSD1_dB(127:189)) mean(Accel_PSD1_dB(190:252)) ...
                 mean(Accel_PSD1_dB(253:316))];
AreaUE2 = 0.5 * [mean(Accel_PSD2_dB(1:63)) mean(Accel_PSD2_dB(64:126)) ...
                 mean(Accel_PSD2_dB(127:189)) mean(Accel_PSD2_dB(190:252)) ...
                 mean(Accel_PSD2_dB(253:316))];
AreaUE3 = 0.5 * [mean(Accel_PSD3_dB(1:63)) mean(Accel_PSD3_dB(64:126)) ...
                 mean(Accel_PSD3_dB(127:189)) mean(Accel_PSD3_dB(190:252)) ...
                 mean(Accel_PSD3_dB(253:316))];
AreaUE4 = 0.5 * [mean(Accel_PSD4_dB(1:63)) mean(Accel_PSD4_dB(64:126)) ...
                 mean(Accel_PSD4_dB(127:189)) mean(Accel_PSD4_dB(190:252)) ...
                 mean(Accel_PSD4_dB(253:316))];
             
cumuDiff_UE1_UE2 = abs(sum(AreaUE1-AreaUE2));  
cumuDiff_UE3_UE4 = abs(sum(AreaUE3-AreaUE4));  
cumuDiff_UE1_UE3 = abs(sum(AreaUE1-AreaUE3));  
cumuDiff_UE2_UE3 = abs(sum(AreaUE2-AreaUE3));  


% % NMSE_UE1_UE2 = abs(immse(Accel_PSD1_dB,Accel_PSD2_dB) / (mean(Accel_PSD1_dB)*mean(Accel_PSD2_dB)));
% % NMSE_UE3_UE4 = abs(immse(Accel_PSD3_dB,Accel_PSD4_dB) / (mean(Accel_PSD3_dB)*mean(Accel_PSD4_dB)));
% % NMSE_UE1_UE3 = abs(immse(Accel_PSD1_dB,Accel_PSD3_dB) / (mean(Accel_PSD1_dB)*mean(Accel_PSD3_dB)));
% % NMSE_UE2_UE3 = abs(immse(Accel_PSD2_dB,Accel_PSD3_dB) / (mean(Accel_PSD2_dB)*mean(Accel_PSD3_dB)));
% % Mean_PSD_dB = [mean(Accel_PSD1_dB) mean(Accel_PSD2_dB) mean(Accel_PSD3_dB) mean(Accel_PSD4_dB)];
% % NMSE = [NMSE_UE1_UE2 NMSE_UE3_UE4 NMSE_UE1_UE3 NMSE_UE2_UE3];


freq = Fs*(0:(N/2))/N;
% 
figure(1)
sgtitle('Power Spectral Density of Acceleration')
subplot(221)
sub1 = plot(freq,Accel_PSD1_dB,freq,Accel_PSD2_dB);
legend('UE1','UE2')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
title(['UE1 vs UE2   Area difference = ', num2str(cumuDiff_UE1_UE2)],'FontWeight','normal')
sub1(1).Color = UE1_color;
sub1(1).LineWidth = line_width;
sub1(2).Color = UE2_color;
sub1(2).LineWidth = line_width;

subplot(222)
sub2 = plot(freq,Accel_PSD3_dB,freq,Accel_PSD4_dB);
legend('UE3','UE4')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
title(['UE3 vs UE4   Area difference = ', num2str(cumuDiff_UE3_UE4)],'FontWeight','normal')
sub2(1).Color = UE3_color;
sub2(1).LineWidth = line_width;
sub2(2).Color = UE4_color;
sub2(2).LineWidth = line_width;

subplot(223)
sub3 = plot(freq,Accel_PSD1_dB,freq,Accel_PSD3_dB);
legend('UE1','UE3')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
title(['UE1 vs UE3   Area difference = ', num2str(cumuDiff_UE1_UE3)],'FontWeight','normal')
sub3(1).Color = UE1_color;
sub3(1).LineWidth = line_width;
sub3(2).Color = UE3_color;
sub3(2).LineWidth = line_width;


subplot(224)
sub4 = plot(freq,Accel_PSD2_dB,freq,Accel_PSD3_dB);
legend('UE2','UE3')
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
title(['UE2 vs UE3   Area difference = ', num2str(cumuDiff_UE2_UE3)],'FontWeight','normal')
sub4(1).Color = UE2_color;
sub4(1).LineWidth = line_width;
sub4(2).Color = UE3_color;
sub4(2).LineWidth = line_width;


%%% Gyroscope Record %%%

GyroUE1 = readRecord(fid_gyro_record_UE1);
GyroUE2 = readRecord(fid_gyro_record_UE2);
GyroUE3 = readRecord(fid_gyro_record_UE3);
GyroUE4 = readRecord(fid_gyro_record_UE4);


GyroUE1_temp = ArrangeDataAndTime(GyroUE1);
GyroUE2_temp = ArrangeDataAndTime(GyroUE2);
GyroUE3_temp = ArrangeDataAndTime(GyroUE3);
GyroUE4_temp = ArrangeDataAndTime(GyroUE4);


Gyro_newT1 = GyroUE1_temp(:,1);
Gyro_newT2 = GyroUE2_temp(:,1);
Gyro_newT3 = GyroUE3_temp(:,1);
Gyro_newT4 = GyroUE4_temp(:,1);

Gyro_T = unique([Gyro_newT1' Gyro_newT2' Gyro_newT3' Gyro_newT4']);
newGyro_T = 0:t_interval:Gyro_T(end); 

newGyroUE1 = dataInterpolation(newGyro_T,Gyro_newT1,GyroUE1_temp);
newGyroUE2 = dataInterpolation(newGyro_T,Gyro_newT2,GyroUE2_temp);
newGyroUE3 = dataInterpolation(newGyro_T,Gyro_newT3,GyroUE3_temp);
newGyroUE4 = dataInterpolation(newGyro_T,Gyro_newT4,GyroUE4_temp);

GyroUE1_sample = sampleByTime(newGyroUE1,t_interval);
GyroUE2_sample = sampleByTime(newGyroUE2,t_interval);
GyroUE3_sample = sampleByTime(newGyroUE3,t_interval);
GyroUE4_sample = sampleByTime(newGyroUE4,t_interval);

GyroUE1_sum = sqrt(GyroUE1_sample(:,2).^2 + GyroUE1_sample(:,3).^2 + GyroUE1_sample(:,4).^2 );
GyroUE2_sum = sqrt(GyroUE2_sample(:,2).^2 + GyroUE2_sample(:,3).^2 + GyroUE2_sample(:,4).^2 );
GyroUE3_sum = sqrt(GyroUE3_sample(:,2).^2 + GyroUE3_sample(:,3).^2 + GyroUE3_sample(:,4).^2 );
GyroUE4_sum = sqrt(GyroUE4_sample(:,2).^2 + GyroUE4_sample(:,3).^2 + GyroUE4_sample(:,4).^2 );

GyroUE1_net = diff(GyroUE1_sum);
GyroUE2_net = diff(GyroUE2_sum);
GyroUE3_net = diff(GyroUE3_sum);
GyroUE4_net = diff(GyroUE4_sum);

% 
% %%% PSD %%%
n = 4; % x:2, y:3, z:4

Gyro_R1_temp = xcorr(GyroUE1_sample(:,n));
Gyro_R2_temp = xcorr(GyroUE2_sample(:,n));
Gyro_R3_temp = xcorr(GyroUE3_sample(:,n));
Gyro_R4_temp = xcorr(GyroUE4_sample(:,n));

GyroUE1_AutoCor = xcorr(GyroUE1_sum);
GyroUE2_AutoCor = xcorr(GyroUE2_sum);
GyroUE3_AutoCor = xcorr(GyroUE3_sum);
GyroUE4_AutoCor = xcorr(GyroUE4_sum);

Gyro_length = length(Gyro_R1_temp);

x = (1 + length(Gyro_R1_temp)) / 2 ;
% x = length(Gyro_R1_temp);
Gyro_R1 = Gyro_R1_temp(1:x)' ./ (1:x) ;
Gyro_R2 = Gyro_R2_temp(1:x)' ./ (1:x) ;
Gyro_R3 = Gyro_R3_temp(1:x)' ./ (1:x) ;
Gyro_R4 = Gyro_R4_temp(1:x)' ./ (1:x) ;

GyroUE1_Cor = GyroUE1_AutoCor(1:x)' ./ (1:x);
GyroUE2_Cor = GyroUE2_AutoCor(1:x)' ./ (1:x);
GyroUE3_Cor = GyroUE3_AutoCor(1:x)' ./ (1:x);
GyroUE4_Cor = GyroUE4_AutoCor(1:x)' ./ (1:x);
% 
% N = length(Gyro_R1);
% 
% %%% auto 
% Gyro_FFT1 = fft(Gyro_R1);
% Gyro_FFT1 = Gyro_FFT1(1:N/2+1);
% Gyro_FFT2 = fft(Gyro_R2);
% Gyro_FFT2 = Gyro_FFT2(1:N/2+1);
% Gyro_FFT3 = fft(Gyro_R3);
% Gyro_FFT3 = Gyro_FFT3(1:N/2+1);
% Gyro_FFT4 = fft(Gyro_R4);
% Gyro_FFT4 = Gyro_FFT4(1:N/2+1);
% 
% Gyro_FFT1 = fft(GyroUE1_Cor);
% Gyro_FFT1 = Gyro_FFT1(1:N/2+1);
% Gyro_FFT2 = fft(GyroUE2_Cor);
% Gyro_FFT2 = Gyro_FFT2(1:N/2+1);
% Gyro_FFT3 = fft(GyroUE3_Cor);
% Gyro_FFT3 = Gyro_FFT3(1:N/2+1);
% Gyro_FFT4 = fft(GyroUE4_Cor);
% Gyro_FFT4 = Gyro_FFT4(1:N/2+1);
% 
% Gyro_PSD1 = (1/(Fs*N)) * abs(Gyro_FFT1);
% Gyro_PSD1(2:end-1) = 2 * Gyro_PSD1(2:end-1);
% Gyro_PSD1_dB = 10*log10(Gyro_PSD1);
% Gyro_PSD2 = (1/(Fs*N)) * abs(Gyro_FFT2);
% Gyro_PSD2(2:end-1) = 2 * Gyro_PSD2(2:end-1) ;
% Gyro_PSD2_dB = 10*log10(Gyro_PSD2);
% Gyro_PSD3 = (1/(Fs*N)) * abs(Gyro_FFT3);
% Gyro_PSD3(2:end-1) = 2 * Gyro_PSD3(2:end-1) ;
% Gyro_PSD3_dB = 10*log10(Gyro_PSD3);
% Gyro_PSD4 = (1/(Fs*N)) * abs(Gyro_FFT4);
% Gyro_PSD4(2:end-1) = 2 * Gyro_PSD4(2:end-1) ;
% Gyro_PSD4_dB = 10*log10(Gyro_PSD4);
% 
% AreaUE1 = 0.5 * [mean(Gyro_PSD1_dB(1:63)) mean(Gyro_PSD1_dB(64:126)) ...
%                  mean(Gyro_PSD1_dB(127:189)) mean(Gyro_PSD1_dB(190:252)) ...
%                  mean(Gyro_PSD1_dB(253:315))];
% AreaUE2 = 0.5 * [mean(Gyro_PSD2_dB(1:63)) mean(Gyro_PSD2_dB(64:126)) ...
%                  mean(Gyro_PSD2_dB(127:189)) mean(Gyro_PSD2_dB(190:252)) ...
%                  mean(Gyro_PSD2_dB(253:315))];
% AreaUE3 = 0.5 * [mean(Gyro_PSD3_dB(1:63)) mean(Gyro_PSD3_dB(64:126)) ...
%                  mean(Gyro_PSD3_dB(127:189)) mean(Gyro_PSD3_dB(190:252)) ...
%                  mean(Gyro_PSD3_dB(253:315))];
% AreaUE4 = 0.5 * [mean(Gyro_PSD4_dB(1:63)) mean(Gyro_PSD4_dB(64:126)) ...
%                  mean(Gyro_PSD4_dB(127:189)) mean(Gyro_PSD4_dB(190:252)) ...
%                  mean(Gyro_PSD4_dB(253:315))];             
% cumuDiff_UE1_UE2 = abs(sum(AreaUE1-AreaUE2));  
% cumuDiff_UE3_UE4 = abs(sum(AreaUE3-AreaUE4));  
% cumuDiff_UE1_UE3 = abs(sum(AreaUE1-AreaUE3));  
% cumuDiff_UE2_UE3 = abs(sum(AreaUE2-AreaUE3));  
% % 
% % 
% % 
% % 
% % 
% % NMSE_UE1_UE2 = abs(immse(Gyro_PSD1_dB,Gyro_PSD2_dB) / (mean(Gyro_PSD1_dB)*mean(Gyro_PSD2_dB)));
% % NMSE_UE3_UE4 = abs(immse(Gyro_PSD3_dB,Gyro_PSD4_dB) / (mean(Gyro_PSD3_dB)*mean(Gyro_PSD4_dB)));
% % NMSE_UE1_UE3 = abs(immse(Gyro_PSD1_dB,Gyro_PSD3_dB) / (mean(Gyro_PSD1_dB)*mean(Gyro_PSD3_dB)));
% % NMSE_UE2_UE3 = abs(immse(Gyro_PSD2_dB,Gyro_PSD3_dB) / (mean(Gyro_PSD2_dB)*mean(Gyro_PSD3_dB)));
% % Mean_PSD_dB = [mean(Gyro_PSD1_dB) mean(Gyro_PSD2_dB) mean(Gyro_PSD3_dB) mean(Gyro_PSD4_dB)]
% % NMSE = [NMSE_UE1_UE2 NMSE_UE3_UE4 NMSE_UE1_UE3 NMSE_UE2_UE3]
% % 
% % 
% freq = Fs*(0:(N/2))/N;
% 
% figure(1)
% sgtitle('Power Spectral Density of Angular Velocity')
% subplot(221)
% sub1 = plot(freq,Gyro_PSD1_dB,freq,Gyro_PSD2_dB);
% title(['UE1 vs UE2  Area difference = ', num2str(cumuDiff_UE1_UE2)],'FontWeight','normal')
% legend('UE1','UE2')
% xlabel('Frequency (Hz)')
% ylabel('Power (dB)')
% sub1(1).Color = UE1_color;
% sub1(1).LineWidth = line_width;
% sub1(2).Color = UE2_color;
% sub1(2).LineWidth = line_width;
% 
% subplot(222)
% sub2 = plot(freq,Gyro_PSD3_dB,freq,Gyro_PSD4_dB);
% title(['UE3 vs UE4  Area difference = ', num2str(cumuDiff_UE3_UE4)],'FontWeight','normal')
% legend('UE3','UE4')
% xlabel('Frequency (Hz)')
% ylabel('Power (dB)')
% sub2(1).Color = UE3_color;
% sub2(1).LineWidth = line_width;
% sub2(2).Color = UE4_color;
% sub2(2).LineWidth = line_width;
% 
% subplot(223)
% sub3 = plot(freq,Gyro_PSD1_dB,freq,Gyro_PSD3_dB);
% title(['UE1 vs UE3  Area difference = ', num2str(cumuDiff_UE1_UE3)],'FontWeight','normal')
% legend('UE1','UE3')
% xlabel('Frequency (Hz)')
% ylabel('Power (dB)')
% sub3(1).Color = UE1_color;
% sub3(1).LineWidth = line_width;
% sub3(2).Color = UE3_color;
% sub3(2).LineWidth = line_width;
% 
% 
% subplot(224)
% sub4 = plot(freq,Gyro_PSD2_dB,freq,Gyro_PSD3_dB);
% title(['UE2 vs UE3  Area difference = ', num2str(cumuDiff_UE2_UE3)],'FontWeight','normal')
% legend('UE2','UE3')
% xlabel('Frequency (Hz)')
% ylabel('Power (dB)')
% sub4(1).Color = UE2_color;
% sub4(1).LineWidth = line_width;
% sub4(2).Color = UE3_color;
% sub4(2).LineWidth = line_width;

% % % 
% figure(2)
% suptitle('Gyroscope X-axis Coherence')
% subplot(221)
% plot(f12,Gyro_C12)
% xlabel('Frequency (Hz)')
% ylabel('Coherence')
% legend('UE1 & UE2')
% subplot(222)
% plot(f34,Gyro_C34)
% xlabel('Frequency (Hz)')
% ylabel('Coherence')
% legend('UE3 & UE4')
% subplot(223)
% plot(f13,Gyro_C13)
% xlabel('Frequency (Hz)')
% ylabel('Coherence')
% legend('UE1 & UE3')
% subplot(224)
% plot(f23,Gyro_C23)
% xlabel('Frequency (Hz)')
% ylabel('Coherence')
% legend('UE2 & UE3')

%%% Magnetic Field Records %%%
% MagUE1 = readRecord(fid_mag_record_UE1);
% MagUE2 = readRecord(fid_mag_record_UE2);
% MagUE3 = readRecord(fid_mag_record_UE3);
% MagUE4 = readRecord(fid_mag_record_UE4);
% MagUE1_record = str2double(MagUE1(:,3:5));
% MagUE2_record = str2double(MagUE2(:,3:5));
% MagUE3_record = str2double(MagUE3(:,3:5));
% MagUE4_record = str2double(MagUE4(:,3:5));
% 
% 
% MagUE1_FFT = fft(MagUE1_record);%,Accel_Sample);
% MagUE2_FFT = fft(MagUE2_record);%,Accel_Sample);
% MagUE3_FFT = fft(MagUE3_record);%,Accel_Sample);
% MagUE4_FFT = fft(MagUE4_record);%,Accel_Sample);
% [MagUE1_FFT_SIZE, ~] = size(MagUE1_FFT(:,1));
% [MagUE2_FFT_SIZE, ~] = size(MagUE2_FFT(:,1));
% [MagUE3_FFT_SIZE, ~] = size(MagUE3_FFT(:,1));
% [MagUE4_FFT_SIZE, ~] = size(MagUE4_FFT(:,1));
% Mag_Fs = 250; % sampling rate
% Mag_f1 = Mag_Fs*(0:(MagUE1_FFT_SIZE/2))/MagUE1_FFT_SIZE;
% Mag_f2 = Mag_Fs*(0:(MagUE2_FFT_SIZE/2))/MagUE2_FFT_SIZE;
% Mag_f3 = Mag_Fs*(0:(MagUE3_FFT_SIZE/2))/MagUE3_FFT_SIZE;
% Mag_f4 = Mag_Fs*(0:(MagUE4_FFT_SIZE/2))/MagUE4_FFT_SIZE;
% 
% axis_n = 3; % adjust axis
% MagUE1_record_X = MagUE1_record(:,axis_n);
% MagUE2_record_X = MagUE2_record(:,axis_n);
% MagUE3_record_X = MagUE3_record(:,axis_n);
% MagUE4_record_X = MagUE4_record(:,axis_n);
% MagUE1_FFT_X = MagUE1_FFT(:,axis_n);
% MagUE2_FFT_X = MagUE2_FFT(:,axis_n);
% MagUE3_FFT_X = MagUE3_FFT(:,axis_n);
% MagUE4_FFT_X = MagUE4_FFT(:,axis_n);
% 
% MagUE1_P2 = abs(MagUE1_FFT_X /MagUE1_FFT_SIZE );
% MagUE1_P1 = MagUE1_P2(1:MagUE1_FFT_SIZE/2+1);
% MagUE1_P1(2:end-1) = 2*MagUE1_P1(2:end-1);
% 
% MagUE2_P2 = abs(MagUE2_FFT_X /MagUE2_FFT_SIZE );
% MagUE2_P1 = MagUE2_P2(1:MagUE2_FFT_SIZE/2+1);
% MagUE2_P1(2:end-1) = 2*MagUE2_P1(2:end-1);
% 
% MagUE3_P2 = abs(MagUE3_FFT_X /MagUE3_FFT_SIZE );
% MagUE3_P1 = MagUE3_P2(1:MagUE3_FFT_SIZE/2+1);
% MagUE3_P1(2:end-1) = 2*MagUE3_P1(2:end-1);
% 
% MagUE4_P2 = abs(MagUE4_FFT_X /MagUE4_FFT_SIZE );
% MagUE4_P1 = MagUE4_P2(1:MagUE4_FFT_SIZE/2+1);
% MagUE4_P1(2:end-1) = 2*MagUE4_P1(2:end-1);
% figure(1)
% suptitle('Time domain Z-axis');
% subplot(221)
% plot(1:MagUE1_FFT_SIZE,MagUE1_record_X,'-',1:MagUE2_FFT_SIZE, MagUE2_record_X, '--')
% legend('UE1','UE2')
% subplot(222)
% plot(1:MagUE3_FFT_SIZE,MagUE3_record_X,'-',1:MagUE4_FFT_SIZE, MagUE4_record_X, '--')
% legend('UE3','UE4')
% subplot(223)
% plot(1:MagUE1_FFT_SIZE,MagUE1_record_X,'-',1:MagUE3_FFT_SIZE, MagUE3_record_X, '--')
% legend('UE1','UE3')
% subplot(224)
% plot(1:MagUE2_FFT_SIZE,MagUE2_record_X,'-',1:MagUE3_FFT_SIZE, MagUE3_record_X, '--')
% legend('UE2','UE3')
% 
% figure(2)
% suptitle('Frequency domain Z-axis');
% subplot(221)
% plot(Mag_f1,MagUE1_P1,'-',Mag_f2,MagUE2_P1, '--')
% legend('UE1','UE2')
% subplot(222)
% plot(Mag_f3,MagUE3_P1,'-',Mag_f4,MagUE4_P1, '--')
% legend('UE3','UE4')
% subplot(223)
% plot(Mag_f1,MagUE1_P1,'-',Mag_f3,MagUE3_P1, '--')
% legend('UE1','UE3')
% subplot(224)
% plot(Mag_f2,MagUE2_P1,'-',Mag_f3,MagUE3_P1, '--')
% legend('UE2','UE3')
% 
% % x=[UE1_P1 UE2_P1];
% mx = [dtw(MagUE1_P1,MagUE2_P1) dtw(MagUE3_P1,MagUE4_P1) dtw(MagUE1_P1,MagUE3_P1) dtw(MagUE2_P1,MagUE3_P1)];


function recordsOutput = readRecord(fid)
    data = textscan(fid,'%s%s%s%s%s%s');
    recordsOutput = horzcat(data{:});
    fclose(fid);
end

function t = t_calculate(UE_time)
SecondsPerDay = 24*60*60;
l = length(UE_time);
    for i = 1:l-1
     t_interval(i) =  (UE_time(i+1) - UE_time(i))*SecondsPerDay;
    end
    t(1) = 0;
    for j = 1:i
     t(j+1) = t(j) + t_interval(j);
    end
end

function UE_all = findUnique(UE_all,t)
[~ ,unq] = unique(UE_all(:,1)); 
id = setdiff(find(t),unq);

for k = 1 : length(id)
    UE_all(id(k)-1,1) = (UE_all(id(k),1) + UE_all(id(k)-1,1))/2;
    UE_all(id(k)-1,2) = (UE_all(id(k),2) + UE_all(id(k)-1,2))/2;
    UE_all(id(k)-1,3) = (UE_all(id(k),3) + UE_all(id(k)-1,3))/2;
    UE_all(id(k)-1,4) = (UE_all(id(k),4) + UE_all(id(k)-1,4))/2;
end
UE_all(id,:) = [];
end

function newUE_all = ArrangeDataAndTime(UE)
    UE_record = str2double(UE(:,3:5));
    UE_time = datenum(UE(:,2), 'HH:MM:SS.FFF');
    UE_exp_t = t_calculate(UE_time);
    UE_all = [UE_exp_t' UE_record];
    newUE_all = findUnique(UE_all,UE_exp_t);
end

function modifyUE_all = dataInterpolation(T,newT,data)
    T_diff = setdiff(T,newT);
    x_data = interp1(newT,data(:,2),T_diff,'linear','extrap');
    y_data = interp1(newT,data(:,3),T_diff,'linear','extrap');
    z_data = interp1(newT,data(:,4),T_diff,'linear','extrap');
    InterpData = [ T_diff ; x_data ; y_data ; z_data]';
    modifyUE_all = sortrows([data ; InterpData]);
end

function sampleUE = sampleByTime(data, time_interval)
id = find(mod(data(:,1),time_interval) == 0);
length(id);

for i = 1 : length(id)
    sampleUE(i,1) = data(id(i),1);
    sampleUE(i,2) = data(id(i),2);
    sampleUE(i,3) = data(id(i),3);
    sampleUE(i,4) = data(id(i),4);
end
end
