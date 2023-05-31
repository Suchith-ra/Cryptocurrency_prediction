#!/usr/bin/env python
# coding: utf-8

# In[40]:


from pymongo import MongoClient


# In[73]:


uri = 'mongodb://localhost:27017/'
client = MongoClient(uri)


# In[74]:


db = client['Cryptocurrency_n']
collection = db['Model_data']


# In[43]:


import import_ipynb


# In[44]:


import Models 


# In[45]:


get_ipython().run_line_magic('store', '-r mse_rf')
get_ipython().run_line_magic('store', '-r rmse_rf')
get_ipython().run_line_magic('store', '-r mae_rf')
get_ipython().run_line_magic('store', '-r r2_score_rf')
get_ipython().run_line_magic('store', '-r adj_r2_rf')
get_ipython().run_line_magic('store', '-r mpe_rf')
get_ipython().run_line_magic('store', '-r mape_rf')
get_ipython().run_line_magic('store', '-r cod_rf')


# In[46]:


get_ipython().run_line_magic('store', '-r mse_xgb')
get_ipython().run_line_magic('store', '-r rmse_xgb')
get_ipython().run_line_magic('store', '-r mae_xgb')
get_ipython().run_line_magic('store', '-r r2_score_xgb')
get_ipython().run_line_magic('store', '-r adj_r2_xgb')
get_ipython().run_line_magic('store', '-r mpe_xgb')
get_ipython().run_line_magic('store', '-r mape_xgb')
get_ipython().run_line_magic('store', '-r cod_xgb')


# In[47]:


get_ipython().run_line_magic('store', '-r mse_lstm')
get_ipython().run_line_magic('store', '-r rmse_lstm')
get_ipython().run_line_magic('store', '-r mae_lstm')
get_ipython().run_line_magic('store', '-r r2_score_lstm')
get_ipython().run_line_magic('store', '-r adj_r2_lstm')
get_ipython().run_line_magic('store', '-r mpe_lstm')
get_ipython().run_line_magic('store', '-r mape_lstm')
get_ipython().run_line_magic('store', '-r cod_lstm')


# In[36]:


get_ipython().run_line_magic('store', '-r mse_bilstm')
get_ipython().run_line_magic('store', '-r rmse_bilstm')
get_ipython().run_line_magic('store', '-r mae_bilstm')
get_ipython().run_line_magic('store', '-r r2_score_bilstm')
get_ipython().run_line_magic('store', '-r adj_r2_bilstm')
get_ipython().run_line_magic('store', '-r mpe_bilstm')
get_ipython().run_line_magic('store', '-r mape_bilstm')
get_ipython().run_line_magic('store', '-r cod_bilstm')


# In[37]:


get_ipython().run_line_magic('store', '-r mse_gru')
get_ipython().run_line_magic('store', '-r rmse_gru')
get_ipython().run_line_magic('store', '-r mae_gru')
get_ipython().run_line_magic('store', '-r r2_score_gru')
get_ipython().run_line_magic('store', '-r adj_r2_gru')
get_ipython().run_line_magic('store', '-r mpe_gru')
get_ipython().run_line_magic('store', '-r mape_gru')
get_ipython().run_line_magic('store', '-r cod_gru')


# In[38]:


image_data_rf = 'https://images1.blob.core.windows.net/projectimages/RF_graph.png'
image_data_xgb = 'https://images1.blob.core.windows.net/projectimages/XGB_graph.png'
image_data_lstm = 'https://images1.blob.core.windows.net/projectimages/LSTM_graph.png'
image_data_bilstm = 'https://images1.blob.core.windows.net/projectimages/BiLSTM_graph.png'
image_data_gru = 'https://images1.blob.core.windows.net/projectimages/GRU_graph.png'


# In[48]:


import pandas as pd 
csv_file = 'C:/Users/suchi/Downloads/dataset (1).csv'
data = pd.read_csv(csv_file)


# In[49]:


data_dict = data.to_dict(orient='records')


# In[50]:


collection.insert_many(data_dict)


# In[66]:





# In[ ]:





# In[ ]:





# In[39]:


Model_data = [
    {
        "model_name": 'RF',
        "mse": mse_rf,
        "rmse": rmse_rf,
        "mae": mae_rf,
        "r2_score": r2_score_rf,
        "adj_r2": adj_r2_rf,
        "mpe": mpe_rf,
        "mape":mape_rf,
        "cod":cod_rf, 
        "graph_data": image_data_rf 
    },
    {
        "model_name": "XGB",
        "mse": mse_xgb,
        "rmse": rmse_xgb,
        "mae": mae_xgb,
        "r2_score": r2_score_xgb,
        "adj_r2": adj_r2_xgb,
        "mpe": mpe_xgb,
        "mape":mape_xgb,
        "cod":cod_xgb,
        "graph_data": image_data_xgb
    },
    {
        "model_name": "LSTM",
        "mse": mse_lstm,
        "rmse": mse_lstm,
        "mae": mae_xgb,
        "r2_score": r2_score_lstm,
        "adj_r2": adj_r2_lstm,
        "mpe": mpe_lstm,
        "mape":mape_lstm,
        "cod":cod_lstm,
        "graph_data": image_data_lstm
    },
    {
        "model_name": "BiLSTM",
        "mse": mse_bilstm,
        "rmse": mse_bilstm,
        "mae": mae_bilstm,
        "r2_score": r2_score_bilstm,
        "adj_r2": adj_r2_bilstm,
        "mpe": mpe_bilstm,
        "mape":mape_bilstm,
        "cod":cod_bilstm,
        "graph_data": image_data_bilstm
    },
    {
        "model_name": "GRU",
        "mse": mse_gru,
        "rmse": mse_gru,
        "mae": mae_gru,
        "r2_score": r2_score_gru,
        "adj_r2": adj_r2_gru,
        "mpe": mpe_gru,
        "mape":mape_gru,
        "cod":cod_gru,
        "graph_data": image_data_gru
    }
    
]

collection.insert_many(Model_data)


# In[ ]:




