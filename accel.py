import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# def accel_txt_to_csv(txt_name,csv_name):
#     columns = ["date","time","acc_x","acc_y","acc_z"]
#     data = pd.read_csv(txt_name,sep=" ",names=columns)
#     data_txtDF = pd.DataFrame(data)
#     data_txtDF.to_csv(csv_name,index=False)

# for n in range(2):
#     txt_name='accel_record_UE{UE}.txt'.format(UE=n+1)
#     csv_name='accel_record_UE{UE}.csv'.format(UE=n+1)
#     accel_txt_to_csv(txt_name,csv_name)

# df = pd.read_csv('accel_record_UE1.csv',index_col="time") 
df = pd.read_csv('accel_record_UE1.csv',parse_dates=[[0,1]],keep_date_col=True) #讀csv(檔案,合併date&time欄位並解析成time_stamps,合併後保留原本欄位)
df = df.set_index('date_time')  #把time_stamps設為index

df=df.drop_duplicates(['time'], keep='first', inplace=False)    #找出time欄位中重複的資料並刪除，保留第一個出現的
print(df[df.index.duplicated()])    #確認已無重複資料

df = df.resample('5L').mean() #resmaple,S秒,L毫秒
df=df.interpolate(method='time', limit_direction='both')    #內差法 method=(time,nearest,zero,slinear,quadratic,cubic)
print(df)
df.plot.line(linewidth=0.5)
plt.show()