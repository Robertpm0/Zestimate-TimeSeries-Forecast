
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas_datareader as web
import seaborn as sns
from luminaire.optimization.hyperparameter_optimization import HyperparameterOptimization
from luminaire.exploration.data_exploration import  DataExploration
import talib
'''
data  = pd.read_csv('ccvol.csv',header=0,names=['date','vol'])
print(data)
data.index = data.date
data = data.drop('date',axis=1)
print(data)
df =  data.rename({'vol':'raw'},axis=1)
'''





coin = str('ETH')
f = requests.get(f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin}&tsym=USD&limit=2000").json()['Data']['Data']
g = pd.DataFrame(f)
df = g[['time','close','volumeto','high','low']]
df['time'] = pd.to_datetime(df['time'], unit='s')
df.index = df['time']
df = df.drop('time',axis=1)
df = df.rename({'close':'raw'},axis=1)
df['smix'] = (df['raw'] / df['raw'].rolling(45).mean()) / (df['raw'].rolling(15).mean() / df['raw'].rolling(45).mean())
df['mayer'] = df['raw'] / df['raw'].rolling(180).mean()
df['rolling'] = df['raw'].rolling(60).mean()
df['rolling2'] = df['raw'].rolling(200).mean()
df['rolling3'] = df['raw'].rolling(7).mean()
df['rsi'] = talib.RSI(df['raw'],14)
df['macd'],df['macd2'],df['macd3'] = talib.MACD(df['raw'],fastperiod=12,slowperiod=26,signalperiod=9)
df.dropna(inplace=True)
print(df)


de = DataExploration(freq='D')
fill = de.add_missing_index(df=df,freq='D')
print(fill)
print(fill.index.min(),fill.index.max())

op = HyperparameterOptimization(freq='D')
opt =  op.run(df)

print(opt)
deOpt = DataExploration(freq='D', **opt)

df2, edit = deOpt.profile(df)

print(df2)




print(df2.index.min(),df2.index.max())
className = opt['LuminaireModel']
moduleName = __import__('luminaire.model', fromlist=[''])
moduleClass = getattr(moduleName, className)
print(moduleClass)

#moduleObject = LADStructuralModel(hyper_params=opt, freq='D')

moduleObject = moduleClass(hyper_params=opt,freq='D')



succ, dte, dfTrained = moduleObject.train(data=df2, **edit)


print(succ,dte,dfTrained)
scr = dfTrained.score(1000,'2022-09-2')
print(scr)
