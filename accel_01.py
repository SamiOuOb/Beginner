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

#gyro_record_UE1
def gyro_txt_to_csv(txt_name,csv_name,n):
    x="UE{UE}_gyro_x".format(UE=n+1)
    y="UE{UE}_gyro_y".format(UE=n+1)
    z="UE{UE}_gyro_z".format(UE=n+1)
    columns = ["date","time",x,y,z]
    data = pd.read_csv(txt_name,sep=" ",names=columns)
    data_txtDF = pd.DataFrame(data)
    data_txtDF.to_csv(csv_name,index=False)

def NMSE(xfp,xfp_1):
    NMSE=0
    aver_1=0
    aver_2=0
    count=0
    expect_value=0
    for i in range(min(len(xfp),len(xfp_1))-1):
        i=i+1
        aver_1+=xfp[i]
        aver_2+=xfp_1[i]
        expect_value+=(xfp[i]-xfp_1[i])**2
        count=count+1
    NMSE=abs((expect_value/count)/((aver_1/count)*(aver_2/count)))
    return NMSE
    
def fft(signal):
    sampling_rate = len(signal)      #采样率
    fft_size = len(signal)      #FFT长度
    t = np.arange(0, 1.0, 1.0/sampling_rate)
    x = signal
    xs = x[:fft_size]
    xf = np.fft.rfft(xs) / fft_size  #返回fft_size/2+1 个频率  
    freqs = np.linspace(0, sampling_rate/2, fft_size/2+1) /len(signal)*2  #表示频率
    xfp = np.log10(np.abs(xf) * 2)*10    #db
    return xfp

# for n in range(4):
#     txt_name='accel_record_UE{UE}.txt'.format(UE=n+1)
#     csv_name='accel_record_UE{UE}.csv'.format(UE=n+1)
#     accel_txt_to_csv(txt_name,csv_name,n)

# for n in range(4):
#     txt_name='gyro_record_UE{UE}.txt'.format(UE=n+1)
#     csv_name='gyro_record_UE{UE}.csv'.format(UE=n+1)
#     gyro_txt_to_csv(txt_name,csv_name,n)

df_all=pd.DataFrame()
df_all_gyro=pd.DataFrame()
for n in range(4): 
    csv_name='accel_record_UE{UE}.csv'.format(UE=n+1)
    df = pd.read_csv(csv_name,parse_dates=[[0,1]],keep_date_col=True) #讀csv(檔案,合併date&time欄位並解析成time_stamps,合併後保留原本欄位)
    df = df.set_index('date_time')  #把time_stamps設為index

    df=df.drop_duplicates(['time'], keep='first', inplace=False)    #找出time欄位中重複的資料並刪除，保留第一個出現的
    # print(df[df.index.duplicated()])    #確認已無重複資料
    df=df.resample('5L').mean() #resmaple,S秒,L毫秒
    df=df.interpolate(method='quadratic', limit_direction='both')    #內差法 method=(time,nearest,zero,slinear,quadratic,cubic)
    df_all=pd.merge(df_all,df, left_index=True, right_index=True, how='outer')

for n in range(4): 
    csv_name='gyro_record_UE{UE}.csv'.format(UE=n+1)
    df_gyro = pd.read_csv(csv_name,parse_dates=[[0,1]],keep_date_col=True) #讀csv(檔案,合併date&time欄位並解析成time_stamps,合併後保留原本欄位)
    df_gyro = df_gyro.set_index('date_time')  #把time_stamps設為index

    df_gyro=df_gyro.drop_duplicates(['time'], keep='first', inplace=False)    #找出time欄位中重複的資料並刪除，保留第一個出現的
    # print(df[df.index.duplicated()])    #確認已無重複資料
    df_gyro=df_gyro.resample('5L').mean() #resmaple,S秒,L毫秒
    df_gyro=df_gyro.interpolate(method='quadratic', limit_direction='both')    #內差法 method=(time,nearest,zero,slinear,quadratic,cubic)
    df_all_gyro=pd.merge(df_all_gyro,df_gyro, left_index=True, right_index=True, how='outer')

# print(df_all.shape[0])
# print(df_all.shape[1])

#消除頭尾雜訊
# total_rows = len(df.index)
# df = df[int(total_rows * data_scale):-int(total_rows * data_scale)]

start=int(df_all.shape[0]*(10/100))
end=int(df_all.shape[0]*(80/100))
df_all=df_all[start:end]

start=int(df_all_gyro.shape[0]*(10/100))
end=int(df_all_gyro.shape[0]*(80/100))
df_all_gyro=df_all_gyro[start:end]

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
df_all['UE2_acc_all']=(df_all['UE2_acc_x']**2+df_all['UE2_acc_y']**2+df_all['UE2_acc_z']**2)**0.5
df_all['UE3_acc_all']=(df_all['UE3_acc_x']**2+df_all['UE3_acc_y']**2+df_all['UE3_acc_z']**2)**0.5
df_all['UE4_acc_all']=(df_all['UE4_acc_x']**2+df_all['UE4_acc_y']**2+df_all['UE4_acc_z']**2)**0.5
signal1 = np.array(autocorr(df_all['UE1_acc_all']))
signal2 = np.array(autocorr(df_all['UE2_acc_all']))
signal3 = np.array(autocorr(df_all['UE3_acc_all']))
signal4 = np.array(autocorr(df_all['UE4_acc_all']))
signal1x = np.array(autocorr(df_all['UE1_acc_x']))
signal2x = np.array(autocorr(df_all['UE2_acc_x']))
signal3x = np.array(autocorr(df_all['UE3_acc_x']))
signal4x = np.array(autocorr(df_all['UE4_acc_x']))
signal1y = np.array(autocorr(df_all['UE1_acc_y']))
signal2y = np.array(autocorr(df_all['UE2_acc_y']))
signal3y = np.array(autocorr(df_all['UE3_acc_y']))
signal4y = np.array(autocorr(df_all['UE4_acc_y']))
signal1z = np.array(autocorr(df_all['UE1_acc_z']))
signal2z = np.array(autocorr(df_all['UE2_acc_z']))
signal3z = np.array(autocorr(df_all['UE3_acc_z']))
signal4z = np.array(autocorr(df_all['UE4_acc_z']))

df_all_gyro['UE1_gyro_all']=(df_all_gyro['UE1_gyro_x']**2+df_all_gyro['UE1_gyro_y']**2+df_all_gyro['UE1_gyro_z']**2)**0.5
df_all_gyro['UE2_gyro_all']=(df_all_gyro['UE2_gyro_x']**2+df_all_gyro['UE2_gyro_y']**2+df_all_gyro['UE2_gyro_z']**2)**0.5
df_all_gyro['UE3_gyro_all']=(df_all_gyro['UE3_gyro_x']**2+df_all_gyro['UE3_gyro_y']**2+df_all_gyro['UE3_gyro_z']**2)**0.5
df_all_gyro['UE4_gyro_all']=(df_all_gyro['UE4_gyro_x']**2+df_all_gyro['UE4_gyro_y']**2+df_all_gyro['UE4_gyro_z']**2)**0.5
signal1_g = np.array(autocorr(df_all_gyro['UE1_gyro_all']))
signal2_g = np.array(autocorr(df_all_gyro['UE2_gyro_all']))
signal3_g = np.array(autocorr(df_all_gyro['UE3_gyro_all']))
signal4_g = np.array(autocorr(df_all_gyro['UE4_gyro_all']))
signal1x_g = np.array(autocorr(df_all_gyro['UE1_gyro_x']))
signal2x_g = np.array(autocorr(df_all_gyro['UE2_gyro_x']))
signal3x_g = np.array(autocorr(df_all_gyro['UE3_gyro_x']))
signal4x_g = np.array(autocorr(df_all_gyro['UE4_gyro_x']))
signal1y_g = np.array(autocorr(df_all_gyro['UE1_gyro_y']))
signal2y_g = np.array(autocorr(df_all_gyro['UE2_gyro_y']))
signal3y_g = np.array(autocorr(df_all_gyro['UE3_gyro_y']))
signal4y_g = np.array(autocorr(df_all_gyro['UE4_gyro_y']))
signal1z_g = np.array(autocorr(df_all_gyro['UE1_gyro_z']))
signal2z_g = np.array(autocorr(df_all_gyro['UE2_gyro_z']))
signal3z_g = np.array(autocorr(df_all_gyro['UE3_gyro_z']))
signal4z_g = np.array(autocorr(df_all_gyro['UE4_gyro_z']))



# xfp1=fft(signal1)
# xfp2=fft(signal2)
# xfp3=fft(signal3)
# xfp4=fft(signal4)
# print(NMSE(xfp1,xfp2))

df_NMSE = pd.DataFrame()
n=100  #組數
for i in range(n):
    xfp1=fft(signal1[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2=fft(signal2[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3=fft(signal3[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4=fft(signal4[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp1x=fft(signal1x[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2x=fft(signal2x[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3x=fft(signal3x[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4x=fft(signal4x[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp1y=fft(signal1y[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2y=fft(signal2y[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3y=fft(signal3y[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4y=fft(signal4y[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp1z=fft(signal1z[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2z=fft(signal2z[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3z=fft(signal3z[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4z=fft(signal4z[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])

    xfp1_g=fft(signal1_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2_g=fft(signal2_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3_g=fft(signal3_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4_g=fft(signal4_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp1x_g=fft(signal1x_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2x_g=fft(signal2x_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3x_g=fft(signal3x_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4x_g=fft(signal4x_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp1y_g=fft(signal1y_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2y_g=fft(signal2y_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3y_g=fft(signal3y_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4y_g=fft(signal4y_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp1z_g=fft(signal1z_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp2z_g=fft(signal2z_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp3z_g=fft(signal3z_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])
    xfp4z_g=fft(signal4z_g[i*(int(len(signal1)/n)):(i+1)*(int(len(signal1)/n))])

    # print(NMSE(xfp,xfp_1),'\n') 
    s = pd.Series({'acc_all':NMSE(xfp1,xfp2),'acc_x':NMSE(xfp1x,xfp2x),'acc_y':NMSE(xfp1y,xfp2y),'acc_z':NMSE(xfp1z,xfp2z),
        'gyro_all':NMSE(xfp1_g,xfp2_g),'gyro_x':NMSE(xfp1x_g,xfp2x_g),'gyro_y':NMSE(xfp1y_g,xfp2y_g),'gyro_z':NMSE(xfp1z_g,xfp2z_g),'UE1vsUE2': 1,'on_same': 1})
    df_NMSE = df_NMSE.append(s, ignore_index=True)
    s = pd.Series({'acc_all':NMSE(xfp3,xfp4),'acc_x':NMSE(xfp3x,xfp4x),'acc_y':NMSE(xfp3y,xfp4y),'acc_z':NMSE(xfp3z,xfp4z),
        'gyro_all':NMSE(xfp3_g,xfp4_g),'gyro_x':NMSE(xfp3x_g,xfp4x_g),'gyro_y':NMSE(xfp3y_g,xfp4y_g),'gyro_z':NMSE(xfp3z_g,xfp4z_g),'UE3vsUE4': 1,'on_same': 1})
    df_NMSE = df_NMSE.append(s, ignore_index=True)
    s = pd.Series({'acc_all':NMSE(xfp1,xfp3),'acc_x':NMSE(xfp1x,xfp3x),'acc_y':NMSE(xfp1y,xfp3y),'acc_z':NMSE(xfp1z,xfp3z),
        'gyro_all':NMSE(xfp1_g,xfp3_g),'gyro_x':NMSE(xfp1x_g,xfp3x_g),'gyro_y':NMSE(xfp1y_g,xfp3y_g),'gyro_z':NMSE(xfp1z_g,xfp3z_g),'UE1vsUE3': 1,'on_same': 0})
    df_NMSE = df_NMSE.append(s, ignore_index=True)
    s = pd.Series({'acc_all':NMSE(xfp2,xfp3),'acc_x':NMSE(xfp2x,xfp3x),'acc_y':NMSE(xfp2y,xfp3y),'acc_z':NMSE(xfp2z,xfp3z),
        'gyro_all':NMSE(xfp2_g,xfp3_g),'gyro_x':NMSE(xfp2x_g,xfp3x_g),'gyro_y':NMSE(xfp2y_g,xfp3y_g),'gyro_z':NMSE(xfp2z_g,xfp3z_g),'UE2vsUE3': 1,'on_same': 0})
    df_NMSE = df_NMSE.append(s, ignore_index=True)

# print(df_NMSE)
df_NMSE.to_csv('NMSE.csv',index=False,sep=',',na_rep=0) 
# print(len(df_all_gyro))
# print(len(df_all))
df_NMSE.plot()
plt.show()


# plt.plot( freqs_1,xfp_1,'b')
# plt.plot( freqs,xfp,'g')
# plt.xlabel(u"Hz", fontproperties='FangSong')
# plt.ylabel(u'db', fontproperties='FangSong')
# plt.subplots_adjust(hspace=0.4)
# plt.show()
