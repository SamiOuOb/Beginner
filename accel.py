import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import seaborn
import math


#UE1&UE2 in A；UE3&UE4 in B；UE4 in backpack


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size // 2:]

def accel_txt_to_csv(txt_name,csv_name,n):
    x="UE{UE}_acc_x".format(UE=n+1)
    y="UE{UE}_acc_y".format(UE=n+1)
    z="UE{UE}_acc_z".format(UE=n+1)
    columns = ["date","time",x,y,z]
    data = pd.read_csv(txt_name,sep=" ",names=columns)
    data_txtDF = pd.DataFrame(data)
    data_txtDF.to_csv(csv_name,index=False)

# for n in range(4):
#     txt_name='accel_record_UE{UE}.txt'.format(UE=n+1)
#     csv_name='accel_record_UE{UE}.csv'.format(UE=n+1)
#     accel_txt_to_csv(txt_name,csv_name,n)

df_all=pd.DataFrame()
for n in range(4): 
    csv_name='accel_record_UE{UE}.csv'.format(UE=n+1)
    df = pd.read_csv(csv_name,parse_dates=[[0,1]],keep_date_col=True) #讀csv(檔案,合併date&time欄位並解析成time_stamps,合併後保留原本欄位)
    df = df.set_index('date_time')  #把time_stamps設為index

    df=df.drop_duplicates(['time'], keep='first', inplace=False)    #找出time欄位中重複的資料並刪除，保留第一個出現的
    print(df[df.index.duplicated()])    #確認已無重複資料
    df=df.resample('5L').mean() #resmaple,S秒,L毫秒
    df=df.interpolate(method='quadratic', limit_direction='both')    #內差法 method=(time,nearest,zero,slinear,quadratic,cubic)
    df_all=pd.merge(df_all,df, left_index=True, right_index=True, how='outer')

# print(df_all.shape[0])
# print(df_all.shape[1])

#消除頭尾雜訊
# total_rows = len(df.index)
# df = df[int(total_rows * data_scale):-int(total_rows * data_scale)]

start=int(df_all.shape[0]*(10/100))
end=int(df_all.shape[0]*(80/100))
df_all=df_all[start:end]

# acc_x=df_all.plot(y='UE1_acc_x',linewidth=0.5)
# df_all.plot(ax=acc_x,y='UE2_acc_x',linewidth=0.5)
# df_all.plot(ax=acc_x,y='UE3_acc_x',linewidth=0.5)
# df_all.plot(ax=acc_x,y='UE4_acc_x',linewidth=0.5)

# acc_y=df_all.plot(y='UE1_acc_y',linewidth=0.5)
# df_all.plot(ax=acc_y,y='UE2_acc_y',linewidth=0.5)
# df_all.plot(ax=acc_y,y='UE3_acc_y',linewidth=0.5)
# df_all.plot(ax=acc_y,y='UE4_acc_y',linewidth=0.5)

# acc_z=df_all.plot(y='UE1_acc_z',linewidth=0.5)
# df_all.plot(ax=acc_z,y='UE2_acc_z',linewidth=0.5)
# df_all.plot(ax=acc_z,y='UE3_acc_z',linewidth=0.5)
# df_all.plot(ax=acc_z,y='UE4_acc_z',linewidth=0.5)

# ax2=df_all.plot(linewidth=0.5)
# plt.show()


# fftResult = np.abs(np.fft.fft(signal)) 
# fftResult_log = (np.log10(np.abs(np.fft.fft(signal))))*10
# PSD=fftResult/len(signal)*5
# PSD_log=(np.log10(PSD))*10
# n = signal.size

# freq = np.fft.fftfreq(n, d = 1)

# # print(fftResult)
# # print(freq)

# plt.plot(1/freq[:int(len(fftResult)/2)]/len(signal),PSD_log[:int(len(fftResult)/2)],linewidth=0.5) 
# # print(fftResult_log)
# plt.show()

df_all['UE1_acc_all']=(df_all['UE1_acc_x']**2+df_all['UE1_acc_y']**2+df_all['UE1_acc_z']**2)**0.5
signal = np.array(autocorr(df_all['UE1_acc_x']))

sampling_rate = len(signal)      #采样率
fft_size = len(signal)      #FFT长度
t = np.arange(0, 1.0, 1.0/sampling_rate)
x = signal
xs = x[:fft_size]
xf = np.fft.rfft(xs) / fft_size  #返回fft_size/2+1 个频率  
freqs = np.linspace(0, sampling_rate/2, fft_size/2+1) /len(signal)*2  #表示频率
xfp = np.log10(np.abs(xf) * 2)*10    #db

plt.plot(freqs , xfp)
plt.xlabel(u"Hz", fontproperties='FangSong')
plt.ylabel(u'db', fontproperties='FangSong')
plt.subplots_adjust(hspace=0.4)
plt.show()