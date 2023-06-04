import pickle
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import numpy as np


def lime_classify_largemodel(X_in):
    model = pickle.load(open("E:\Thesis_app\model\model_100_boosted_forest_2.pkl", "rb"))
    training_data = pd.read_csv("E:/Thesis_app/data/training_classify_large_data.csv")

    unnamed_cols  =  training_data.columns.str.contains('Unnamed')
    training_data = training_data.drop(training_data[training_data.columns[unnamed_cols]], axis=1)
    
    
    X = training_data
    X = X.drop(columns=["range"], axis = 1)

    explainer = LimeTabularExplainer(np.array(X), 
                                   feature_names=X.columns, 
                                   class_names=['0','1','2','3','4'], 
                                   verbose=True, 
                                   mode='classification')
    
    exp = explainer.explain_instance(X_in.iloc[0], 
                                 model.predict_proba, 
                                 num_features=10)
    
    html_exp = exp.as_html()
    return html_exp