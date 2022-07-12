import tensorflow_hub as hub

import pandas as pd

import tensorflow_text as text

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

import numpy as np

import random

from azureml.core.workspace import Workspace
from azureml.core.run import Run
#from azureml.core.dataset import Dataset
from azureml.core.model import Model as azure_model



ws = Workspace(
    subscription_id = "88ffd436-6b2f-4a5c-942f-72cc66d5bfab",
    resource_group = "aarav_resources",
    workspace_name = "aarav_workspace",
)


#loading dataset


data=pd.read_csv('spam.csv')

df_ham_downsampled=df_ham.sample(len(df_spam))

df_balanced = pd.concat([df_spam , df_ham_downsampled])

df_balanced=df_balanced.sort_index()

df_balanced['spam'] = df_balanced['Category'].apply(lambda x:1 if x=='spam' else 0)

X_train, X_test , y_train, y_test = train_test_split(df_balanced['Message'], df_balanced['spam'],test_size=0.2)

bert_preprocessor = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
bert_encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4')

text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'Inputs')
preprocessed_text = bert_preprocessor(text_input)
embeed = bert_encoder(preprocessed_text)
dropout = tf.keras.layers.Dropout(0.1, name = 'Dropout')(embeed['pooled_output'])
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'Dense')(dropout)

model = tf.keras.Model(inputs = [text_input], outputs = [outputs])

model.summary()

Metrics = [tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),
           tf.keras.metrics.Precision(name = 'precision'),
           tf.keras.metrics.Recall(name = 'recall')
           ]

# compiling our model
model.compile(optimizer ='adam',
               loss = 'binary_crossentropy',
               metrics = Metrics)

run = Run.get_context()

history = model.fit(X_train, y_train, epochs = 2)

# Evaluating performance
# model.evaluate(X_test,y_test)

# getting y_pred by predicting over X_text and flattening it
# y_pred = model.predict(X_test)
# y_pred = y_pred.flatten() # require to be in one-dimensional array , for easy manipulation

import os
bert_path=os.path.join('','model_save')
os.makedirs(bert_path,exist_ok=True)
model.save(os.path.join(bert_path,'full_model_email.h5'))
azure_model.register(workspace=ws,model_path=bert_path + "/full_model.h5",model_name="spambert")
run.complete()
#-to save pickle file -->
#import joblib
#joblib.dump(model, 'email_spam.pkl')

