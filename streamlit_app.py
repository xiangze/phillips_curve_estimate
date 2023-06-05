#!/usr/bin/env python
# coding: utf-8

# # 統計データからフィリップス曲線を推定する2023年版

# 他にもわかりそうなことがあれば解析する

# In[1]:
import streamlit as st
pp=st.plotly_chart

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from colorspacious import cspace_converter

import stan


# In[3]:


from datetime import datetime
from datetime import timedelta


# # 消費者物価指数

# https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200573&tstat=000001150147&cycle=0&tclass1=000001150151&tclass2=000001150152&tclass3=000001150153&tclass4=000001150156&stat_infid=000032103842&tclass5val=0

# In[8]:


df=pd.read_csv("https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032103842&fileKind=1",encoding='shift_jis',index_col='類・品目', parse_dates=True)
#df=pd.read_csv("zmi2020s.csv",encoding='shift_jis',index_col='類・品目', parse_dates=True)
cpi=df[5:]
cpi=cpi.astype("float")
cpi


# 1970年からデータがある

# In[52]:


pp( px.line(cpi,title='消費者物価指数')
)


# In[48]:


pp( px.line(cpi, y="総合", title='消費者物価指数(総合)')
)


# In[50]:


cpi.columns


# ### 日付変換

# In[189]:


pd.to_datetime([s[:4]+"-"+s[4:] for s in cpi.index])
cpi.index=pd.to_datetime([s[:4]+"-"+s[4:] for s in cpi.index])
#pd.to_datetime(cpi.index)


# In[192]:

pp(
    px.line(cpi,title='消費者物価指数')
)


# In[233]:


pp(
    px.line(cpi.iloc[:,:10],title='消費者物価指数(総合)')
)


# 教養娯楽用耐久財と家庭用耐久財の価格下落は何？

# # 失業率

# 最初のシート(デフォルト)は季節調整値で２つ目が原数値

# In[129]:


unemploy_org=pd.read_excel("https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a10.xlsx", sheet_name="原数値")


# ### 注意書き

# In[122]:


unemploy_note=pd.read_excel("https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a10.xlsx", sheet_name="※注_Notes")
unemploy_note


# In[134]:


unemploy=unemploy_org
unemploy[:14]


# 原数値は沖縄返還後の1972年7月からデータがある

# ### 列名の作成

# In[136]:


import itertools

u1=unemploy[unemploy.index == 4].values.flatten()
u2=unemploy[unemploy.index == 6].values.flatten()
print(u1)
print(u2)
u1=list(itertools.chain.from_iterable([[s,s,s] for s in u1[1::3]]))
u1=["年","月","month","non"]+u1[3:]
print(u1)
unemploy_col=[ str(s[0])+str(s[1]).replace("nan","") for s in zip(u1,u2) ]
unemploy_col
unemploy.columns=unemploy_col
unemploy[:14]


# #### 頭と後ろ(注)を取り去る

# In[143]:


unemploy=unemploy[9:-4]


# #### 年, 月の設定、数値化

# In[153]:


years=unemploy["年"][1::12]
unemploy["年"]=list(itertools.chain.from_iterable([[s]*12 for s in years]))
unemploy


# ### 全部floatにする(雑)

# In[217]:


unemploy["月"]=[s.replace("月","") for s in  unemploy["月"].values]
unemploy["month"]=unemploy["月"]
#unemploy.index=unemploy["年"]+unemploy["月"]
unemploy.index=pd.to_datetime(unemploy["年"].astype(str)+"-"+unemploy["月"].astype(str))
unemploy=unemploy.drop(["年","月","month","non"],axis=1)


# In[229]:


unemploy_population=unemploy.replace("… ","0").astype("float").iloc[:,:-3]
unemploy_rate=unemploy.replace("… ","0").astype("float").iloc[:,-3:]

pp( px.line(unemploy_population)
)

pp( px.line(unemploy_rate)
)


# # 賃金

# In[ ]:


# https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00450091&tstat=000001011429


# 反映が遅い？

# # CPIと失業率のMerge

# In[292]:


cpi["総合物価上昇率(年率)"]= np.concatenate(([1]*12 ,(cpi["総合"][12:].values/cpi["総合"][:-12]).values))


# In[293]:


cpi["総合物価上昇率(年率)"]


# In[294]:


pp( px.line(cpi["総合物価上昇率(年率)"])
)


# In[295]:


philips=pd.merge(cpi["総合物価上昇率(年率)"],unemploy_rate["完全失業率（％）男女計"],left_index=True, right_index=True)
philips["date"]=philips.index
philips["year"]=[str(s)[:4] for s in philips.index]


# In[297]:


pp( px.scatter(philips,x="総合物価上昇率(年率)",y="完全失業率（％）男女計",color="year")
)


# In[ ]:




