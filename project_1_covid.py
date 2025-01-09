#%%
from time_series_helper import WindowGenerator

#%%
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
print(keras.backend.backend())

# %%
import mlflow
import datetime
import numpy as np
import pandas as pd
import mlflow.sklearn
import seaborn as sns
import matplotlib as mpl
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from keras import layers, metrics, losses, optimizers, activations, Sequential, Input, Model, regularizers, callbacks

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# %%
TEST_CSV_PATH= os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_test.csv')
test = pd.read_csv(TEST_CSV_PATH)

# %%
TRAIN_CSV_PATH= os.path.join(os.getcwd(), 'dataset', 'cases_malaysia_train.csv')
train = pd.read_csv(TRAIN_CSV_PATH)

# %% EDA & DATA INSPECTING FOR TEST DATA
test.head()

# %%
test.info()

# %%
print(test.isna().sum())

# %%
date = test['date']
test = test.drop('date', axis=1)

# %%
for i in test:
    test[i] = test[i].replace(' ', np.nan)
    test[i] = test[i].replace('?', np.nan)

# %%There is one nan values in dataset. Thus i'm using KNN imputer method to replace nan values instead of drop the missing values row.
columns_name = test.columns #to extract the columns names 

knn_i = KNNImputer()
test = knn_i.fit_transform(test) #convert to a numpy array
test = pd.DataFrame(test) # convert back into data frame
test.columns = columns_name # to put back the columns name
print(test)

# %%
print(test.isna().sum())

# %%
plot_cols = ['cases_new', 'cases_import', 'cases_recovered', 'cases_active', 'cases_cluster', 'cases_unvax', 'cases_pvax', 'cases_fvax', 'cases_boost',	'cases_child',	'cases_adolescent',	'cases_adult',	'cases_elderly', 'cases_0_4', 'cases_5_11',	'cases_12_17',	'cases_18_29', 'cases_30_39', 'cases_40_49', 'cases_50_59',	'cases_60_69', 'cases_70_79', 'cases_80', 'cluster_import',	'cluster_religious', 'cluster_community', 'cluster_highRisk', 'cluster_education',	'cluster_detentionCentre',	'cluster_workplace']

plot_features = test[plot_cols]
plot_features.index = date
_ = plot_features.plot(subplots=True)

plot_features = test[plot_cols][:480]
plot_features.index = date[:480]
_ = plot_features.plot(subplots=True)

#%%
test.describe().transpose()

# %%EDA & DATA INSPECTING FOR TRAIN DATA
train.head()

# %%
train.info()

# %%
train.describe()

# %%
print(train.isna().sum())

# %%
date = train['date']
train = train.drop('date', axis=1)

# %%
for i in train:
    train[i] = train[i].replace(' ', np.nan)
    train[i] = train[i].replace('?', np.nan)
    
# %% cannot remove row that have nan values becsause almost half of the dataset in train file nan values. Thus, i'm using KNN imputer to replcae nan values.
columns_name = train.columns

knn_i = KNNImputer()
train = knn_i.fit_transform(train) 
train = pd.DataFrame(train) 
train.columns = columns_name 
print(train)

#%%
print(train.isna().sum())

#%%
train[['cases_new']] = train[['cases_new']].astype('float64')

#%%
train.info()

# %%
plot_features = train[plot_cols]
plot_features.index = date
_ = plot_features.plot(subplots=True)

plot_features = train[plot_cols][:480]
plot_features.index = date[:480]
_ = plot_features.plot(subplots=True)

# %%
train.describe().transpose()

# %%
train.shape
#not spliting for validation because the trauin dataset did not have many data for train data.

#%%
train_size = train.shape[0]

train_ratio = 0.9
val_ratio = 0.1

train_dt = train[:int(train_ratio*train_size)]
val = train[:int(val_ratio*train_size)]

# %%
train_mean = train.mean()
train_std = train.std()

train_data = (train - train_mean) / train_std
val_data = (val - train_mean) / train_std
test_data = (test - train_mean) / train_std
 
#%%
train_data.shape

#%%
mlflow.set_experiment("covid19 forecast")

# %%
window_1 = WindowGenerator(30,30, 1, train_data, val_data, test_data, label_columns=["cases_new"])
window_1.plot(plot_col="cases_new")

# %%
model_single= tf.keras.Sequential()
model_single.add(tf.keras.layers.LSTM(64, input_shape=(30,30), return_sequences=True))
model_single.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True)) 
# activation relu more lesser than tanh avtivation for mae and loss
model_single.add(tf.keras.layers.Dense(units=1))
model_single.summary()

# %%
model_single.compile(optimizer='adam', loss='mse', metrics=['mae'])

# %%
with mlflow.start_run():
    mlflow.tensorflow.autolog()
    MAX_EPOCHS = 20

    history_single = model_single.fit(window_1.train, validation_data=window_1.val, epochs=MAX_EPOCHS)
# %%
window_1.plot(plot_col="cases_new", model=model_single)

#%%
dir(history_single)


# %%
plt.figure(figsize=(10,10))
plt.plot(history_single.epoch, history_single.history['loss'])
plt.plot(history_single.epoch, history_single.history['val_loss'])
plt.title("")
plt.legend(["Training Loss","Validation Loss"])
plt.show()


# %%
plt.figure(figsize=(10,10))
plt.plot(history_single.epoch, history_single.history['mae'])
plt.plot(history_single.epoch, history_single.history['val_mae'])
plt.title("")
plt.legend(["Training mae","Validation mae"])
plt.show()


# %%
