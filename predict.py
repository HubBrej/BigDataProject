from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MaxAbsScaler
import pandas
import numpy as np
import tensorflow
import keras
from keras.models import load_model


def predict_cl(filename, model):
    #load data
    X_pred = pandas.read_csv(filename)

    religion = X_pred.InsuredInfo_9.unique()

    race = X_pred.InsuredInfo_8.unique()

    print(religion)
    print(race)

    X_pred['InsuredInfo_7'] = X_pred['InsuredInfo_7'].apply(
        lambda x: 0 if x == 'Male' else 1)
    X_pred['InsuredInfo_8'] = X_pred['InsuredInfo_8'].apply(
        lambda x: np.nan
        if type(np.nan) == type(x) else np.where(race == x)[0][0])
    X_pred['InsuredInfo_9'] = X_pred['InsuredInfo_9'].apply(
        lambda x: np.nan
        if type(np.nan) == type(x) else np.where(religion == x)[0][0])
    X_pred['Product_Info_2'] = X_pred['Product_Info_2'].apply(
        lambda x: int(x, 16))

    cols = ['InsuredInfo_7', 'InsuredInfo_8', 'InsuredInfo_9']

    for col1 in cols:

        cr = X_pred[X_pred.columns[1:]].corr()[col1].abs() > 0.1
        idxs = X_pred[X_pred.columns[1:]].corr()[col1][cr].index
        vals = X_pred[col1].unique()
        for col in idxs:
            if col != col1:
                if col != 'Response':
                    for idx in vals:
                        if idx is np.nan:
                            X_pred.loc[X_pred[col1] == idx, col] = X_pred.loc[
                                X_pred[col1].isnull(),
                                col] - X_pred.loc[X_pred[col1].isnull(),
                                                  col].mean()
                        else:
                            X_pred.loc[X_pred[col1] == idx, col] = X_pred.loc[
                                X_pred[col1] == idx,
                                col] - X_pred.loc[X_pred[col1] == idx,
                                                  col].mean()

    simp = SimpleImputer()
    Ximp_pred = simp.fit_transform(X_pred)

    scaler_maxabs = MaxAbsScaler()
    Xscaledimp_pred = scaler_maxabs.fit_transform(Ximp_pred)

    # load the saved model
    # predict classes

    preds = model.predict(Xscaledimp_pred)
    X_pred['Result'] = np.argmax(preds, axis=1)

    X_pred.to_csv('results.csv')
    return preds