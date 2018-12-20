import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#UE1,UE2 in A；UE3,UE4 in B；UE4 in backpack


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
start=int(df_all.shape[0]*(10/100))
end=int(df_all.shape[0]*(80/100))
df_all=df_all[start:end]

acc_x=df_all.plot(y='UE1_acc_x',linewidth=0.5)
df_all.plot(ax=acc_x,y='UE2_acc_x',linewidth=0.5)
df_all.plot(ax=acc_x,y='UE3_acc_x',linewidth=0.5)
df_all.plot(ax=acc_x,y='UE4_acc_x',linewidth=0.5)

acc_y=df_all.plot(y='UE1_acc_y',linewidth=0.5)
df_all.plot(ax=acc_y,y='UE2_acc_y',linewidth=0.5)
df_all.plot(ax=acc_y,y='UE3_acc_y',linewidth=0.5)
df_all.plot(ax=acc_y,y='UE4_acc_y',linewidth=0.5)

acc_z=df_all.plot(y='UE1_acc_z',linewidth=0.5)
df_all.plot(ax=acc_z,y='UE2_acc_z',linewidth=0.5)
df_all.plot(ax=acc_z,y='UE3_acc_z',linewidth=0.5)
df_all.plot(ax=acc_z,y='UE4_acc_z',linewidth=0.5)

ax2=df_all.plot(linewidth=0.5)

plt.show()

