# 統計データ(消費者物価指数、失業率)からフィリップス曲線を推定する2023年版
# 他にもわかりそうなことがあれば解析する
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import stan
import pymc
import itertools
import matplotlib.pyplot as plt
import arviz as az
import nest_asyncio
from datetime import datetime
import shutil
import httpstan.models
import httpstan.cache
import pickle
def clean_model(stan_model):
    try:
        # Get the the name of the folder where your model is saved
        model_name = httpstan.models.calculate_model_name(stan_model)
        print(model_name)
    except:
        print("httpstan is not used")
        return 
    # Then delete the model directory with
    httpstan.cache.delete_model_directory(model_name)
    # Get the path to the model directory in the cache
    model_path_in_cache = httpstan.cache.model_directory(model_name)
    # Finally delete the folder and all the files it contains with shutils
    shutil.rmtree(model_path_in_cache)

#位の平均、標準偏差の図保存
def topmeanvariance(summary,suffix,topvalue=20,figsize=800):
    summary=summary.sort_values(by="mean")
    summary=summary.filter(like='L', axis=0)
    px.bar(summary[-topvalue:], y='名前', x='mean',orientation="h", width=figsize, height=figsize).savefig("img/char_indivisual_mean_20"+suffix+".png")
    px.bar(summary[-topvalue:], y='名前', x='sd',orientation="h", width=figsize, height=figsize).savefig("img/char_indivisual_sd_20"+suffix+".png")

#stan実行結果の保存
def save_results(suffix,fit):
    with open('fit_'+suffix+'.pkl', 'wb') as w:
        pickle.dump(fit, w)
    print("fin")
    #実行結果全体の保存
    fit.to_frame().to_csv("postdata/posterior_"+suffix+".csv") 
    #サマリーの保存
    summary = az.summary(fit)
    summary.to_csv("postdata/summary_"+suffix+".csv") 
    #グラフ描画、保存
    visdata = az.from_pystan(posterior=fit)
    #trace plot
    az.plot_trace(visdata)
    plt.savefig("img/posterior_charm_trace_"+suffix+".png")
    #forest plot
    az.plot_forest(visdata)
    plt.savefig("img/posterior_charm_"+suffix+".png")
    return summary

# 消費者物価指数
cpi_filename="https://www.e-stat.go.jp/stat-search/file-download?statInfId=000032103842&fileKind=1"
# https://www.e-stat.go.jp/stat-search/files?page=1&layout=datalist&toukei=00200573&tstat=000001150147&cycle=0&tclass1=000001150151&tclass2=000001150152&tclass3=000001150153&tclass4=000001150156&stat_infid=000032103842&tclass5val=0
df=pd.read_csv(cpi_filename,encoding='shift_jis',index_col='類・品目', parse_dates=True)
cpi=df[5:].astype("float") #

#  日付変換
pd.to_datetime([s[:4]+"-"+s[4:] for s in cpi.index])
cpi.index=pd.to_datetime([s[:4]+"-"+s[4:] for s in cpi.index])
# 1970年からデータがある
fig = px.line(cpi, y="総合", title='消費者物価指数(総合)')
#fig=go.Figure()
fig.write_image("cpi_general.png")
sepn=20
fig = px.line(cpi.iloc[:,:sepn],title='消費者物価指数(最初の方)')
fig = px.line(cpi.iloc[:,sepn:],title='消費者物価指数(最後)')
fig.write_image("cpi.png")

# 教養娯楽用耐久財と家庭用耐久財の価格下落は何？
# 失業率
# 最初のシート(デフォルト)は季節調整値で２つ目が原数値
unemploy=pd.read_excel("https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a10.xlsx", sheet_name="原数値")
### 注意書き
unemploy_note=pd.read_excel("https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a10.xlsx", sheet_name="※注_Notes")
unemploy_note

# 原数値は沖縄返還後の1972年7月からデータがある
### 列名の作成
u1=unemploy[unemploy.index == 4].values.flatten()
u2=unemploy[unemploy.index == 6].values.flatten()
u1=list(itertools.chain.from_iterable([[s,s,s] for s in u1[1::3]]))
u1=["年","月","month","non"]+u1[3:]
unemploy_col=[ str(s[0])+str(s[1]).replace("nan","") for s in zip(u1,u2) ]
unemploy.columns=unemploy_col

#### 頭と後ろ(注)を取り去る
unemploy=unemploy[9:-4]
#### 年, 月の設定、数値化
years=unemploy["年"][1::12]
unemploy["年"]=list(itertools.chain.from_iterable([[s]*12 for s in years]))
unemploy["月"]=[s.replace("月","") for s in  unemploy["月"].values]
unemploy["month"]=unemploy["月"]
unemploy.index=pd.to_datetime(unemploy["年"].astype(str)+"-"+unemploy["月"].astype(str))
unemploy=unemploy.drop(["年","月","month","non"],axis=1)

unemploy_population=unemploy.replace("… ","0").astype("float").iloc[:,:-3]
unemploy_rate=unemploy.replace("… ","0").astype("float").iloc[:,-3:]

px.line(unemploy_population).write_image("unemploy_population.png")
fig=px.line(unemploy_rate).write_image("unemploy_rate.png")

# 失業率(調整後)
unemploy_after=pd.read_excel("https://www.stat.go.jp/data/roudou/longtime/zuhyou/lt01-a10.xlsx")

#列名の作成   
def set_colname(unemploy):
    u1=unemploy[unemploy.index == 4].values.flatten()
    u2=unemploy[unemploy.index == 6].values.flatten()
    u1=list(itertools.chain.from_iterable([[s,s,s] for s in u1[1::3]]))
    u1=["年","月","month","non"]+u1[3:]
    unemploy.columns=[ str(s[0])+str(s[1]).replace("nan","") for s in zip(u1,u2) ]
    return unemploy
    
def set_year(unemploy):
    #年&月
    years=unemploy["年"][1::12]
    years=list(itertools.chain.from_iterable([[s]*12 for s in years]))
    unemploy["年"]=years[:len(unemploy)]
    unemploy["月"]=[s.replace("月","") for s in  unemploy["月"].values]
    unemploy["month"]=unemploy["月"]
    unemploy.index=pd.to_datetime(unemploy["年"].astype(str)+"-"+unemploy["月"].astype(str))
    unemploy=unemploy.drop(["年","月","month","non"],axis=1)
    a=unemploy.replace("… ","0").astype("float")
    unemploy_population=a.iloc[:,:-3]
    unemploy_rate=a.iloc[:,-3:]
    return unemploy_population,unemploy_rate

#頭と後ろを取り去る
unemploy_after_population,unemploy_after_rate=set_year(set_colname(unemploy_after)[9:-12])

#plotly.express を用いたプロット
fig = px.line(unemploy_after_population)
fig = px.line(unemploy_after_rate)
fig = px.line(unemploy_rate,title="調整なし失業率")
fig.show()

# 賃金
# https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00450091&tstat=000001011429
# 反映が遅い？
# CPIと失業率のMerge
cpi["総合物価上昇率(年率)"]= np.concatenate(([1]*12 ,(cpi["総合"][12:].values/cpi["総合"][:-12]).values))
fig = px.line(cpi["総合物価上昇率(年率)"])
fig.show()
philips=pd.merge(cpi["総合物価上昇率(年率)"],unemploy_rate["完全失業率（％）男女計"],left_index=True, right_index=True)
philips["year"]=[str(s)[:4] for s in philips.index]
fig = px.scatter(philips[6:],x="完全失業率（％）男女計",y="総合物価上昇率(年率)",color="year")
fig.show()

philips_after=pd.merge(cpi["総合物価上昇率(年率)"],unemploy_after_rate["完全失業率（％）男女計"],left_index=True, right_index=True)
philips_after=philips_after
philips_after["year"]=[str(s)[:4] for s in philips_after.index]

fig = px.scatter(philips_after[6:],x="完全失業率（％）男女計",y="総合物価上昇率(年率)",color="year")
fig.show()
fig.write_html()

# stanによるモデルフィッティング
nest_asyncio.apply()
# https://discourse.mc-stan.org/t/pystan-3-beta-4-released/18938/2
phillips=pd.read_csv("phillips_2023_raw.csv")[6:]
# πt=βEt[πt+1]−λϕ(ut−unn)
# πt : インフレ率,
# ut  :失業率,
# unt  : 自然失業率,
# β,λ,ϕ  : 係数
# 参考: 賃金版ニューケインジアン・フィリップス曲線に関する実証分析　日米比較
# https://www.boj.or.jp/research/wps_rev/wps_2014/data/wp14j02.pdf
with open("NKWPC.stan") as f:
     model_code=f.read()

clean_model(model_code) #clean previous result
data={"N":len(phillips),
     "pi":phillips["総合物価上昇率(年率)"].to_numpy().astype("float"),
     "u":phillips["完全失業率（％）男女計"].to_numpy().astype("float")
     }
model= stan.build(model_code, data=data, random_seed=4)        
fit_NKWPC = model.sample(num_chains=4, num_samples=2000) 
save_results("simple",fit_NKWPC)
#az.summary(fit_NKWPC)
# (季節調整なしの値を使った場合)は収束しない

# # 失業率調整後のデータを使った場合
# https://stackoverflow.com/questions/71298720/how-to-force-pystan-to-recompile-a-stan-model
clean_model(model_code)
data_after={"N":len(philips_after),
     "pi":philips_after["総合物価上昇率(年率)"].to_numpy().astype("float"),
     "u":philips_after["完全失業率（％）男女計"].to_numpy().astype("float")     }
model_after= stan.build(model_code, data=data_after, random_seed=4)        
fit_after_NKWPC = model_after.sample(num_chains=4, num_samples=2000) 
az.summary(fit_after_NKWPC)





