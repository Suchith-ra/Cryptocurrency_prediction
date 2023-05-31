#!/usr/bin/env python
# coding: utf-8

# In[15]:


pip install pandas 


# In[3]:


import pandas as pd


# In[3]:


path = "C:/Users/suchi/Downloads/dataset (1).csv"
df = pd.read_csv(path)
df


# In[4]:


df1 = df
df1.info()


# In[5]:



time_coinapi = pd.to_datetime(df1['time_coinapi'], format='%M:%S.%f')
time_coinapi_seconds = time_coinapi.dt.minute * 60 + time_coinapi.dt.second + time_coinapi.dt.microsecond / 1e6
df1['time_coinapi_seconds'] = time_coinapi_seconds


time_exchange = pd.to_datetime(df1['time_exchange'], format='%M:%S.%f')
time_exchange_seconds = time_exchange.dt.minute * 60 + time_exchange.dt.second + time_exchange.dt.microsecond / 1e6
df1['time_exchange_seconds'] = time_exchange_seconds

ts = pd.to_datetime(df1['ts'], format='%M:%S.%f')
ts_seconds = ts.dt.minute * 60 + ts.dt.second + ts.dt.microsecond / 1e6
df1['ts'] = ts_seconds

print(df1)


# In[6]:


df1


# In[23]:


pip install matplotlib


# In[7]:


import matplotlib.dates as dates
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker 

fig,ax=plt.subplots()
ax.plot(df1['time_exchange_seconds'],df1['bid_price'])
plt.show()


# In[8]:


df1 =df1.drop(2,axis=0)
df1 =df1.drop(['time_coinapi','time_exchange'],axis=1)
df1


# In[9]:


fig,ax=plt.subplots()
ax.plot(df1['time_exchange_seconds'],df1['bid_price'])
plt.show()


# In[10]:


y_reg= df1[['bid_price','time_exchange_seconds']]
x_reg= df1.drop(['bid_price','id','symbol'],axis=1)
print(x_reg,y_reg)


# In[29]:


pip install seaborn


# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


corr_matrix = x_reg.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()


threshold = 0.5
corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)


print('Selected features:', corr_features)


# In[12]:


x_reg= df1.drop(['id','symbol','sequence','ask_size','bid_size','ts','time_coinapi_seconds','bid_price'],axis=1)
print(x_reg,y_reg)


# In[14]:



x1_reg = x_reg.groupby('time_exchange_seconds', as_index=False).agg({"ask_price":"mean"})
y1_reg = y_reg.groupby('time_exchange_seconds', as_index=False).agg({"bid_price":"mean"})
y1_reg = y1_reg.drop(['time_exchange_seconds'], axis=1)
print(y1_reg)
print(x1_reg)


# In[34]:


'''x1_reg['time_exchange_seconds'] = x1_reg['time_exchange_seconds'].astype(int)
aggregated_per_second = x1_reg.groupby('time_exchange_seconds').mean().reset_index()
aggregated_per_second '''


# In[35]:


'''x1_reg['minute'] = x1_reg['time_exchange_seconds'] // 60
aggregated_per_minute = x1_reg.groupby('minute').mean().reset_index()
aggregated_per_minute '''


# In[41]:


pip install scikit-learn


# In[15]:


from sklearn.model_selection import train_test_split
x_train_reg, x_test_reg, y_train_reg, y_test_reg = train_test_split(x1_reg, y1_reg, test_size=0.4, random_state=42,shuffle=False)
y_test_reg.reset_index(drop=True, inplace=True)
y_test_reg


# In[43]:


pip install numpy


# In[16]:


from sklearn.ensemble import RandomForestRegressor as RFR
import numpy as np


number_of_models = 10


all_rfr_predictions = []

for i in range(number_of_models):
    rfr_regressor = RFR(n_estimators= 100,min_weight_fraction_leaf=0.25,random_state=42)
    rfr_regressor.fit(x_train_reg, y_train_reg.values.ravel())
    rfr_predictions = rfr_regressor.predict(x_test_reg)
    all_rfr_predictions.append(rfr_predictions)
average_predictions_rf = np.mean(all_rfr_predictions, axis=0)
print(average_predictions_rf)


# In[17]:


predictions_rf = pd.DataFrame(average_predictions_rf,columns=['Predicted values RF'])
predictions_rf


# In[18]:


import pickle

with open('rfmodel.pkl', 'wb') as file:
    pickle.dump(rfr_regressor, file)


# In[19]:


from sklearn.ensemble import GradientBoostingRegressor as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time 

number_of_models = 10


all_predictions_xgb = []

for i in range(number_of_models):
    xgb_regressor = xgb(n_estimators=100, max_depth=20,min_weight_fraction_leaf=0.19 ,random_state=42)
    xgb_regressor.fit(x_train_reg, y_train_reg.values.ravel())
    predictions_xgb = xgb_regressor.predict(x_test_reg)
    all_predictions_xgb.append(predictions_xgb)
average_predictions_xgb = np.mean(all_predictions_xgb, axis=0)
print(average_predictions_xgb)


# In[20]:


predictions_xgb = pd.DataFrame(average_predictions_xgb,columns=['Predicted values XGB'])
predictions_xgb


# In[21]:


import pickle

with open('xgbmodel.pkl', 'wb') as file:
    pickle.dump(xgb_regressor, file)


# In[54]:


get_ipython().system('pip install tensorflow')


# In[22]:


df2 = df1[['bid_price','time_exchange_seconds']]
df2


# In[23]:


#minute_resampled_data = df2.bid_price.resample('1Min').mean()
df2 = df.groupby('time_exchange_seconds', as_index=False).agg({"bid_price":"mean"})
df2 = df2.drop(['time_exchange_seconds'], axis=1)
print(df2)


# In[24]:


from sklearn.preprocessing import MinMaxScaler
data = df2.values
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

time_steps = 10  


X_dl = []
y_dl = []
for i in range(len(data_scaled) - time_steps):
    X_dl.append(data_scaled[i:i + time_steps, 0])
    y_dl.append(data_scaled[i + time_steps, 0])
X_dl = np.array(X_dl)
y_dl = np.array(y_dl)


train_size = int(len(X_dl) * 0.8)
X_dl_train, X_dl_test = X_dl[:train_size], X_dl[train_size:]
y_dl_train, y_dl_test = y_dl[:train_size], y_dl[train_size:]


X_dl_train = np.reshape(X_dl_train, (X_dl_train.shape[0], X_dl_train.shape[1], 1))
X_dl_test = np.reshape(X_dl_test, (X_dl_test.shape[0], X_dl_test.shape[1], 1))


# In[25]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
# Build the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, 1)))
model_lstm.add(LSTM(units=50))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')


model_lstm.fit(X_dl_train, y_dl_train, epochs=50, batch_size=32)


y_pred_lstm = model_lstm.predict(X_dl_test)


y_pred_lstm_inverse = scaler.inverse_transform(y_pred_lstm)
y_test_lstm_inverse = scaler.inverse_transform(y_dl_test.reshape(-1, 1))


# In[26]:


print(y_pred_lstm_inverse)


# In[27]:


import pickle

with open('model_lstm.pkl', 'wb') as file:
    pickle.dump(xgb_regressor, file)


# In[28]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional


model_bilstm = Sequential()
model_bilstm.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(time_steps, 1)))
model_bilstm.add(Bidirectional(LSTM(units=50)))
model_bilstm.add(Dense(units=1))
model_bilstm.compile(optimizer='adam', loss='mean_squared_error')


model_bilstm.fit(X_dl_train, y_dl_train, epochs=50, batch_size=32)


y_pred_bilstm= model_bilstm.predict(X_dl_test)


y_pred_bilstm_inverse = scaler.inverse_transform(y_pred_bilstm)
y_test_bilstm_inverse = scaler.inverse_transform(y_dl_test.reshape(-1, 1))


# In[29]:


print(y_pred_bilstm_inverse)


# In[30]:


import pickle

with open('model_bilstm.pkl', 'wb') as file:
    pickle.dump(model_bilstm, file)


# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,GRU, Dropout
# Build the GRU model
model_gru=Sequential()
model_gru.add(GRU(32,return_sequences=True,input_shape=(time_steps,1)))
model_gru.add(GRU(32,return_sequences=True))
model_gru.add(GRU(32))
model_gru.add(Dropout(0.20))
model_gru.add(Dense(1))
model_gru.compile(loss='mean_squared_error',optimizer='adam')


model_gru.fit(X_dl_train, y_dl_train, epochs=50, batch_size=32)


y_pred_gru= model_gru.predict(X_dl_test)


y_pred_gru_inverse = scaler.inverse_transform(y_pred_gru)
y_test_gru_inverse = scaler.inverse_transform(y_dl_test.reshape(-1, 1))


# In[32]:


print(y_pred_gru_inverse)


# In[33]:


import pickle
with open('model_gru.pkl', 'wb') as file:
    pickle.dump(model_gru, file)


# In[59]:


#RFMODEL METRICS
from sklearn.metrics import *
import math
mse_rf=mean_squared_error(y_test_reg,average_predictions_rf)
print("mse of RF: ",mse_rf)
rmse_rf=math.sqrt(mse_rf)
print("rmse of RF: ",rmse_rf)
mae_rf=mean_absolute_error(y_test_reg,average_predictions_rf)
print("mae of RF:",mae_rf)
r2_score_rf=r2_score(y_test_reg,average_predictions_rf)
print("r2 score of RF: ",r2_score_rf)
n=len(y_test_reg)
p=x_reg.shape[1]
adj_r2_rf=1-(1-r2_score_rf)*(n-1)/(n-p-1)
print("Adjusted RF score:",adj_r2_rf)
#y_test_reg_m=y_test_reg.values.reshape(-1,1)
#average_predictions_rf_m=average_predictions_rf.values.reshape(-1,1)
mpe_rf=np.mean((y_test_reg_m - average_predictions_rf) / y_test_reg_m) * 100
print("mean percetage error RF:",mpe_rf)
y_test_reg, average_predictions_rf = np.array(y_test_reg), np.array(average_predictions_rf)
mape_rf= np.mean(np.abs((y_test_reg - average_predictions_rf) / y_test_reg)) * 100
print("Mean absolute percentage error RF:",mape_rf)
cod_rf = 1 - (np.sum((y_test_reg - average_predictions_rf) * 2) / np.sum((average_predictions_rf - np.mean(average_predictions_rf)) * 2))
print("Coefficient of Determination RF:",cod_rf)


# In[47]:




# In[60]:


get_ipython().run_line_magic('store', 'mse_rf')
get_ipython().run_line_magic('store', 'rmse_rf')
get_ipython().run_line_magic('store', 'mae_rf')
get_ipython().run_line_magic('store', 'r2_score_rf')
get_ipython().run_line_magic('store', 'adj_r2_rf')
get_ipython().run_line_magic('store', 'mpe_rf')
get_ipython().run_line_magic('store', 'mape_rf')
get_ipython().run_line_magic('store', 'cod_rf')


# In[37]:


import matplotlib.pyplot as plt
plt.plot(y_test_reg[:100], label='Actual')
plt.plot(predictions_rf[:100], label='Predicted')
plt.xlabel('Index')
plt.ylabel('Bid Price')
plt.title('Random forest Regressor: Actual vs Predicted')
plt.legend()
plt.show()


# In[62]:


#XGBMODEL METRICS
from sklearn.metrics import *
import math
mse_xgb=mean_squared_error(y_test_reg,average_predictions_xgb)
print("mse of XGB: ",mse_xgb)
rmse_xgb=math.sqrt(mse_xgb)
print("rmse of XGB: ",rmse_xgb)
mae_xgb=mean_absolute_error(y_test_reg,average_predictions_xgb)
print("mae of XGB:",mae_xgb)
r2_score_xgb=r2_score(y_test_reg,average_predictions_xgb)
print("r2 score of XGB: ",r2_score_rf)
n1=len(y_test_reg)
p1=x_reg.shape[1]
adj_r2_xgb=1-(1-r2_score_xgb)*(n-1)/(n1-p1-1)
print("Adjusted XGB score:",adj_r2_xgb)
#y_test_reg_m=y_test_reg.values.reshape(-1,1)
#average_predictions_rf_m=average_predictions_rf.values.reshape(-1,1)
mpe_xgb=np.mean((y_test_reg_m - average_predictions_xgb) / y_test_reg_m) * 100
print("mean percetage error XGB:",mpe_rf)
y_test_reg, average_predictions_xgb = np.array(y_test_reg), np.array(average_predictions_xgb)
mape_xgb= np.mean(np.abs((y_test_reg - average_predictions_xgb) / y_test_reg)) * 100
print("Mean absolute percentage error XGB:",mape_xgb)
cod_xgb = 1 - (np.sum((y_test_reg - average_predictions_xgb) * 2) / np.sum((average_predictions_xgb - np.mean(average_predictions_xgb)) * 2))
print("Coefficient of Determination XGB:",cod_xgb)


# In[52]:


get_ipython().run_line_magic('store', 'mse_xgb')
get_ipython().run_line_magic('store', 'rmse_xgb')
get_ipython().run_line_magic('store', 'mae_xgb')
get_ipython().run_line_magic('store', 'r2_score_xgb')
get_ipython().run_line_magic('store', 'adj_r2_xgb')
get_ipython().run_line_magic('store', 'mpe_xgb')
get_ipython().run_line_magic('store', 'mape_xgb')
get_ipython().run_line_magic('store', 'cod_xgb')


# In[39]:


#XGBMODEL GRAPH
import matplotlib.pyplot as plt
plt.plot(y_test_reg[:100], label='Actual')
plt.plot(predictions_xgb[:100], label='Predicted')
plt.xlabel('Index')
plt.ylabel('Bid Price')
plt.title('XGBoost Regressor: Actual vs Predicted')
plt.legend()
plt.show()


# In[40]:


#LSTMMODEL METRICS
from sklearn.metrics import *
import math
mse_lstm=mean_squared_error(y_test_lstm_inverse,y_pred_lstm_inverse)
print("mse of LSTM: ",mse_lstm)
rmse_lstm=math.sqrt(mse_lstm)
print("rmse of LSTM: ",rmse_lstm)
mae_lstm=mean_absolute_error(y_test_lstm_inverse,y_pred_lstm_inverse)
print("mae of LSTM:",mae_lstm)
r2_score_lstm=r2_score(y_test_lstm_inverse,y_pred_lstm_inverse)
print("r2 score of LSTM: ",r2_score_lstm)
n=len(y_test_lstm_inverse)
p=X_dl.shape[1]
adj_r2_lstm=1-(1-r2_score_lstm)*(n-1)/(n-p-1)
print("Adjusted LSTM score:",adj_r2_lstm)
#y_test_lstm_inverse_m=y_test_lstm_inverse.values.reshape(-1,1)
#average_predictions_rf_m=y_pred_lstm_inverse.values.reshape(-1,1)
mpe_lstm=np.mean((y_test_lstm_inverse - y_pred_lstm_inverse) / y_test_lstm_inverse) * 100
print("mean percetage error LSTM:",mpe_lstm)
y_test_lstm_inverse, y_pred_lstm_inverse = np.array(y_test_lstm_inverse), np.array(y_pred_lstm_inverse)
mape_lstm= np.mean(np.abs((y_test_lstm_inverse - y_pred_lstm_inverse) / y_test_lstm_inverse)) * 100
print("Mean absolute percentage error LSTM:",mape_lstm)
cod_lstm = 1 - (np.sum((y_test_lstm_inverse - y_pred_lstm_inverse) * 2) / np.sum((y_pred_lstm_inverse - np.mean(y_pred_lstm_inverse)) * 2))
print("Coefficient of Determination LSTM:",cod_lstm)


# In[53]:


get_ipython().run_line_magic('store', 'mse_lstm')
get_ipython().run_line_magic('store', 'rmse_lstm')
get_ipython().run_line_magic('store', 'mae_lstm')
get_ipython().run_line_magic('store', 'r2_score_lstm')
get_ipython().run_line_magic('store', 'adj_r2_lstm')
get_ipython().run_line_magic('store', 'mpe_lstm')
get_ipython().run_line_magic('store', 'mape_lstm')
get_ipython().run_line_magic('store', 'cod_lstm')


# In[41]:


#LSTMMODEL GRAPH
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot( y_test_lstm_inverse[:100], label='Actual')
plt.plot(y_pred_lstm_inverse[:100], label='Predicted')
plt.xlabel('Index')
plt.ylabel('Bid Price')
plt.title('LSTM: Predicted vs Actual')
plt.legend()
plt.show()


# In[42]:


#BILSTMMODEL METRICS
# Calculate the root mean squared error
from sklearn.metrics import *
import math
mse_bilstm=mean_squared_error(y_test_bilstm_inverse,y_pred_bilstm_inverse)
print("mse of BiLSTM: ",mse_bilstm)
rmse_bilstm=math.sqrt(mse_bilstm)
print("rmse of BiLSTM: ",rmse_bilstm)
mae_bilstm=mean_absolute_error(y_test_bilstm_inverse,y_pred_bilstm_inverse)
print("mae of BiLSTM:",mae_bilstm)
r2_score_bilstm=r2_score(y_test_lstm_inverse,y_pred_bilstm_inverse)
print("r2 score of BiLSTM: ",r2_score_bilstm)
n=len(y_test_bilstm_inverse)
p=X_dl.shape[1]
adj_r2_bilstm=1-(1-r2_score_bilstm)*(n-1)/(n-p-1)
print("Adjusted BiLSTM score:",adj_r2_bilstm)
#y_test_lstm_inverse_m=y_test_lstm_inverse.values.reshape(-1,1)
#average_predictions_rf_m=y_pred_lstm_inverse.values.reshape(-1,1)
mpe_bilstm=np.mean((y_test_bilstm_inverse - y_pred_bilstm_inverse) / y_test_bilstm_inverse) * 100
print("mean percetage error BiLSTM:",mpe_bilstm)
y_test_bilstm_inverse, y_pred_bilstm_inverse = np.array(y_test_bilstm_inverse), np.array(y_pred_bilstm_inverse)
mape_bilstm= np.mean(np.abs((y_test_bilstm_inverse - y_pred_bilstm_inverse) / y_test_bilstm_inverse)) * 100
print("Mean absolute percentage error BiLSTM:",mape_lstm)
cod_bilstm = 1 - (np.sum((y_test_bilstm_inverse - y_pred_bilstm_inverse) * 2) / np.sum((y_pred_bilstm_inverse - np.mean(y_pred_bilstm_inverse)) * 2))
print("Coefficient of Determination BiLSTM:",cod_bilstm)


# In[56]:


get_ipython().run_line_magic('store', 'mse_bilstm')
get_ipython().run_line_magic('store', 'rmse_bilstm')
get_ipython().run_line_magic('store', 'mae_bilstm')
get_ipython().run_line_magic('store', 'r2_score_bilstm')
get_ipython().run_line_magic('store', 'adj_r2_bilstm')
get_ipython().run_line_magic('store', 'mpe_bilstm')
get_ipython().run_line_magic('store', 'mape_bilstm')
get_ipython().run_line_magic('store', 'cod_bilstm')


# In[43]:


#BILSTMMODEL GRAPH
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot( y_test_bilstm_inverse[:100], label='Actual')
plt.plot(y_pred_bilstm_inverse[:100], label='Predicted')
plt.xlabel('Index')
plt.ylabel('Bid Price')
plt.title('BILSTM: Predicted vs Actual')
plt.legend()
plt.show()


# In[44]:


#GRUMODEL METRICS
# Calculate the root mean squared error
from sklearn.metrics import *
import math
mse_gru=mean_squared_error(y_test_gru_inverse,y_pred_gru_inverse)
print("mse of GRU: ",mse_gru)
rmse_gru=math.sqrt(mse_gru)
print("rmse of GRU: ",rmse_gru)
mae_gru=mean_absolute_error(y_test_gru_inverse,y_pred_gru_inverse)
print("mae of GRU:",mae_gru)
r2_score_gru=r2_score(y_test_gru_inverse,y_pred_gru_inverse)
print("r2 score of GRU: ",r2_score_gru)
n=len(y_test_gru_inverse)
p=X_dl.shape[1]
adj_r2_gru=1-(1-r2_score_gru)*(n-1)/(n-p-1)
print("Adjusted GRU score:",adj_r2_gru)
#y_test_lstm_inverse_m=y_test_lstm_inverse.values.reshape(-1,1)
#average_predictions_rf_m=y_pred_lstm_inverse.values.reshape(-1,1)
mpe_gru=np.mean((y_test_gru_inverse - y_pred_gru_inverse) / y_test_gru_inverse) * 100
print("mean percetage error GRU:",mpe_gru)
y_test_gru_inverse, y_pred_gru_inverse = np.array(y_test_gru_inverse), np.array(y_pred_gru_inverse)
mape_gru= np.mean(np.abs((y_test_gru_inverse - y_pred_gru_inverse) / y_test_gru_inverse)) * 100
print("Mean absolute percentage error GRU:",mape_gru)
cod_gru = 1 - (np.sum((y_test_gru_inverse - y_pred_gru_inverse) * 2) / np.sum((y_pred_gru_inverse - np.mean(y_pred_gru_inverse)) * 2))
print("Coefficient of Determination GRU:",cod_gru)


# In[55]:


get_ipython().run_line_magic('store', 'mse_gru')
get_ipython().run_line_magic('store', 'rmse_gru')
get_ipython().run_line_magic('store', 'mae_gru')
get_ipython().run_line_magic('store', 'r2_score_gru')
get_ipython().run_line_magic('store', 'adj_r2_gru')
get_ipython().run_line_magic('store', 'mpe_gru')
get_ipython().run_line_magic('store', 'mape_gru')
get_ipython().run_line_magic('store', 'cod_gru')


# In[45]:


#GRUMODEL GRAPH
plt.plot( y_test_gru_inverse[:100], label='Actual')
plt.plot( y_pred_gru_inverse[:100], label='Predicted')
plt.xlabel('Index')
plt.ylabel('Bid Price')
plt.title('GRU: Predicted vs Actual')
plt.legend()
plt.show()


# In[ ]:




