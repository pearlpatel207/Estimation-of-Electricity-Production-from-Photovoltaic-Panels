import datetime
import pandas as pd
#processing to combine the several single data files into a single pandas dataframe
irr = pd.read_csv("irradiance.csv", sep = ';')
ac_pow = pd.read_csv("ac_pow.csv", sep = ';')
dc_pow = pd.read_csv("dc_pow.csv", sep = ';')
inv_eff = pd.read_csv("inv_eff.csv", sep = ';')
panel_out = pd.read_csv("panel_out.csv", sep = ';')

irr = irr.drop(['Unnamed: 2', 'Unnamed: 3'], axis = 1)
ac_pow = ac_pow.drop(['Unnamed: 2', 'Unnamed: 3'], axis = 1)
dc_pow = dc_pow.drop(['Unnamed: 2', 'Unnamed: 3'], axis = 1)
inv_eff = inv_eff.drop(['Unnamed: 2', 'Unnamed: 3'], axis = 1)
panel_out = panel_out.drop(['Unnamed: 2', 'Unnamed: 3'], axis = 1)

irr.columns = ['TimeStamp', 'Irr']
ac_pow.columns = ['TimeStamp', 'ac_pow']
dc_pow.columns = ['TimeStamp', 'dc_pow']
inv_eff.columns = ['TimeStamp', 'inv_eff']
panel_out.columns = ['TimeStamp', 'panel_out']

irr['TimeStamp'] = pd.to_datetime(irr['TimeStamp'])
ac_pow['TimeStamp'] = pd.to_datetime(ac_pow['TimeStamp'])
dc_pow['TimeStamp'] = pd.to_datetime(dc_pow['TimeStamp'])
inv_eff['TimeStamp'] = pd.to_datetime(inv_eff['TimeStamp'])
panel_out['TimeStamp'] = pd.to_datetime(panel_out['TimeStamp'])

data = pd.merge_asof(irr,ac_pow, on = 'TimeStamp')
data = pd.merge_asof(data,dc_pow, on = 'TimeStamp')
data = pd.merge_asof(data,panel_out, on = 'TimeStamp')
data = pd.merge_asof(data,inv_eff, on = 'TimeStamp')
data = data.drop(['ac_pow'], axis = 1)

df = data
weather = pd.read_csv("weather_1_1.csv")
df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
df['TimeStamp'] = df['TimeStamp'].apply(lambda x : x.replace(second = 00))
weather['TimeStamp'] = pd.to_datetime(weather['TimeStamp'])
data = pd.merge_asof(df, weather, on = 'TimeStamp')
data.to_csv('PV_data.csv', index = False)
