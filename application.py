#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install flask')


# In[ ]:


get_ipython().system('pip install pyngrok')


# In[15]:


pip install flask-ngrok


# In[1]:


import pandas as pd
from flask import Flask, render_template, request, send_file
from flask_ngrok import run_with_ngrok
from pymongo import MongoClient
app = Flask(__name__,template_folder='C:/Users/suchi/templates_new')
run_with_ngrok(app)


# In[2]:


client = MongoClient('mongodb://localhost:27017')
db = client['Cryptocurrency_n']
collection = db['Model_data']


# In[3]:


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# In[4]:


@app.route('/results', methods=['POST'])
def results():
    model1 = request.form['model1']
    model2 = request.form['model2']

    
    results_model1 = collection.find_one({"model_name": model1})
    results_model2 = collection.find_one({"model_name": model2})

    
    metrics_model1 = {
        'MSE:':results_model1['mse'],
        'RMSE:': results_model1['rmse'],
        'MAE:': results_model1['mae'],
        'R2 Score': results_model1['r2_score'],
        'ADJ_R2':results_model1['adj_r2'],
        'MPE':results_model1['mpe'],
        'MAPE':results_model1['mape'],
        'COD':results_model1['cod']   
    }
    metrics_model2 = {
        'MSE':results_model2['mse'],
        'RMSE': results_model2['rmse'],
        'MAE': results_model2['mae'],
        'R2 Score': results_model2['r2_score'],
        'ADJ_R2':results_model2['adj_r2'],
        'MPE':results_model2['mpe'],
        'MAPE':results_model2['mape'],
        'COD':results_model2['cod']
    }
    

        
        

    img_data1 = results_model1['graph_data']
    img_data2 = results_model2['graph_data']
    
   

   
    return render_template(
        'results.html',
        model1=model1,
        model2=model2,
        metrics_model1=metrics_model1,
        metrics_model2=metrics_model2,
        fig1=img_data1,
        fig2=img_data2
    )


# In[ ]:





# In[ ]:


if __name__ == '__main__':
    app.run()


# In[1]:




# In[4]:




# In[5]:





# In[1]:




# In[2]:





# In[ ]:




