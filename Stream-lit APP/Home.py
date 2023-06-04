import streamlit as st
import pickle
import pandas as pd
import scikitplot as skplt
import matplotlib.pyplot as plt
from protlearn.preprocessing import remove_unnatural
from src import feature_extraction
from src import lime_calculate
import os
import streamlit.components.v1 as components


st.set_page_config(layout="wide")

def load_model():
    model = pickle.load(open("model/model_100_boosted_forest_2.pkl", "rb"))
    return model

def predicted_range_to_value(val):
    val = int(val)
    if val == 0:
        val_min = "0 °C"
        val_max = "20 °C"
        msg = "Topt range 0 min - 20 max"
        return val_min,val_max,msg
    elif val == 1:
        val_min = "20 °C"
        val_max = "40 °C"
        msg = "Topt range 20 min - 40 max"
        return val_min,val_max,msg
    elif val == 2:
        val_min = "40 °C"
        val_max = "60 °C"
        msg = "Topt range 40 min - 60 max"
        return val_min,val_max,msg
    elif val == 3:
        val_min = "60 °C"
        val_max = "80 °C"
        msg = "Topt range 60 min - 80 max"
        return val_min,val_max,msg
    elif val == 4:
        val_min = "80 °C"
        val_max = "120 °C"
        msg = "Topt range 80 min - 120 max"
        return val_min,val_max,msg

def app():
    

    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.text("")
    with col2:
        st.image("static\KU Leuven logo1.png")
    with col3:
        st.text("")
    
    
    col1, col2, col3 = st.columns([1,2,1])
    
    with col1:
        st.text("")
    with col2:
        st.header("Enzyme Catalystic Optmium Temprature Prediction and Infernce Application")
        st.subheader("Enzyme Topt Classification Model: 300 Trees Forest XGBOOST:")
    with col3:
        st.text("")
    
    
    col1, col2, col3 = st.columns([1,2,1])
    
    with col1:
        st.text("")
    with col2:
        seq = st.text_input('Enter Valid Enzyme Sequence','ARKLY')
        try:
            seqs = remove_unnatural(seq)
        except:
            st.error('Please enter a value', icon="🚨")
            exit()

        if len(seq) < 30:
                st.error('Please enter an Enzyme Sequence larger than 30 characters', icon="🚨")
                exit()
        if len(seqs) == 0:
                st.error('Please enter a valid Enzyme Sequence', icon="🚨")
                exit()
    with col3:
        st.text("")
    

    col1, col2, col3 = st.columns([1,2,1])
    
    with col1:
        st.text("")
    with col2:
        button1 = st.button('RUN')
        if button1:
            if len(seq) > 30 and len(seqs) != 0:
                ex_data = feature_extraction.feature_extractions(seq)
                ex_data = ex_data.drop(columns=["seqs"], axis = 1)
                unnamed_cols  =  ex_data.columns.str.contains('Unnamed')
                ex_data = ex_data.drop(ex_data[ex_data.columns[unnamed_cols]], axis=1)
                
                model = load_model()
                
                newval_proba = model.predict_proba(ex_data)
                newval_proba = pd.DataFrame(newval_proba,columns=["vlow","low","med","hi","vhi"])
                
                pred_val = model.predict(ex_data)
                val_min,val_max,msg =predicted_range_to_value(pred_val)
                
                
                my_file = "data/run_ext_data_0.csv"
                if os.path.exists(my_file):
                    os.remove(my_file)

    with col3:
        st.text("")
    


    col1, col2, col3 = st.columns([1,2,1])
    if button1:
        if len(seq) > 30 and len(seqs) != 0:
            with col1:
                st.write("")
            with col2:
                col1_1, col2_1, col3_1 = st.columns([1,2,1])
                with col1_1:
                    st.metric(label= msg, value=val_min)
                    st.metric(label= msg, value=val_max)
                with col2_1:
                    st.image("static\model_AUC_100_BF.png")
                with col3_1:
                    with st.expander("ROC curve Class to Topt Index:"):
                        st.write( "Model Class Performace w.r.t. Temprature range as follows: \n 1. Class 0: Topt range 0 °C - 20 °C \n 2. Class 1 Topt range 20 °C - 40 °C \n 3. Class 2 Topt range 40 °C - 60 °C \n2. Class 3 Topt range 60 °C - 80 °C \n2. Class 4 Topt range > 80 °C  \n")
                    with st.expander("Model Hyperparameter"):
                        st.write("1. Learning Rate: 0.01")
                        st.write("2. No of parallel trees AKA forest: 300")
                        st.write("3. Boosting Algo: Extreme Gradient Boosting")
                        st.write("4. Tree Depth: 10")
                        st.write("5. Data Sub Sampling: 50%")
                        st.write("6. Feature Sub Sampling: 20%")
                    with st.expander("Model Remarks"):
                        st.write("1. Model's Performance in Predicting Class Range from 80 °C to 120 °C is noteworthy")
                        st.write("2. Classification Model's purpose it to provide infernce to the data set")
            with col3:
                st.write("")

    col11, col21, col31 = st.columns([1,2,1])
    with col11:
        st.write("")
    with col21:
        button2 = st.button('RUN Lime Input Inference')
    with col31:
        st.write("")
    
    col12, col22, col32 = st.columns([1,2,1])
    if button2:
        if len(seq) > 30 and len(seqs) != 0:
            with col12:
                st.write("")
            with col22:
                ex_data = feature_extraction.feature_extractions(seq)
                ex_data = ex_data.drop(columns=["seqs"], axis = 1)
                unnamed_cols  =  ex_data.columns.str.contains('Unnamed')
                ex_data = ex_data.drop(ex_data[ex_data.columns[unnamed_cols]], axis=1)
                html_lime = lime_calculate.lime_classify_largemodel(ex_data)
                components.html(html_lime, height=800)
            with col32:
                st.write("")
                
                


if __name__ == "__main__":
    app()
