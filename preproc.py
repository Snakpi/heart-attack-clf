import tensorflow as tf

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from functools import wraps

### Batching them
def create_dataset(bs_train, bs_test, test_size=0.3, return_tensors=False):
    '''
    Create a dataset with batch size
    Output:
    - dl_train (DataLoader obj): dataloader object for training data
    - dl_test (DL obj): dataloader object for testing data (no minibatch, full batch tensor pairs)
    '''

    df = pd.read_csv('data5.csv')
    X = pd.DataFrame()

    ### Splitting features and labels
    for col_name in df.columns.values:
        if col_name not in ['ID', 'Binary Output']:
            X[col_name] = df[col_name]
    if return_tensors:
        y = pd.get_dummies(df['Binary Output'])
    else:
        y = df['Binary Output']

    #### Some fixes
    X.loc[100, 'location'] = 'Delhi'
    loc_types = list(X.location.unique())
    X_ = X.copy(deep=False)

    ### Categorical indexing
    for i in range(len(X)):
        X_.iat[i,1] = 1*(X_.iat[i,1]=='Male')
        X_.iat[i,2] = loc_types.index(X.iat[i,2])

    ## To array
    X_arr = np.array(X_, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, 
        y_arr,
        test_size=test_size, 
        random_state=123,
    )
    
    if return_tensors:
        #y_train = tf.expand_dims(tf.constant(y_train), axis=1)
        #y_test = tf.expand_dims(tf.constant(y_test), axis=1)

        X_train = tf.data.Dataset.from_tensor_slices(X_train).batch(bs_train)
        X_test = tf.data.Dataset.from_tensor_slices(X_test).batch(bs_test)
        y_train = tf.data.Dataset.from_tensor_slices(y_train).batch(bs_train)
        y_test = tf.data.Dataset.from_tensor_slices(y_test).batch(bs_test)
        return X_train, X_test, y_train, y_test
    else: 
        return X_train, X_test, y_train, y_test