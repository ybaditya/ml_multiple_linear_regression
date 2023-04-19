# Importing necessary packages
import streamlit as st
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title='Machine Learning App')
st.title("Machine Learning")
st.subheader("Multiple Linear Regression")
st.write(":blue[version : 00-00.ML.LR.0323]")

st.caption("""Multiple Linear Regression is one of the important regression algorithms which models the linear relationship between a single dependent continuous variable and more than one independent variable.""")
st.caption("The Formula of Multiple Linear Regression : **Y = a1.x1 + a2.x2 + ... + an.xn + b**")
uploaded_file = st.file_uploader('Choose a XLSX file', type='xlsx')
st.caption("_please make sure there is no Null-data, null data will make machine learning error_")
if uploaded_file:
    st.markdown('---')
    data = pd.read_excel(uploaded_file, engine='openpyxl')
    st.write("**Data Preview :**")
    st.write(data.head())
    st.write("**Data Information :**")
    st.caption("""**count** : The number of not-empty values. 
                  ,**mean** : The average (mean) value.
                  ,**std** : The standard deviation.
                  ,**min** : the minimum value.
                  ,**25%** : The 25% percentile*.
                  ,**50%** : The 50% percentile*.
                  ,**75%** : The 75% percentile*.
                  ,**max** : the maximum value.""")
    describe = data.describe()
    st.write(describe)

    st.write("**Corelation Between Data :**")
    st.caption("_Note: **1** is Perfect corelation, **0,9-06** Good Corelation, **<0,5** Bad Corelation, use this data to select Independent Variables_")
    correlation = data.corr()
    st.write(correlation)

    st.write("**Select The Target Variable (_Y_):**")
    target_var = st.selectbox("Target Variable", list(data.columns))

    st.write("**Select The Independent Variables (_X1, X2, X3, ..._):**")
    independent_vars = st.multiselect("Independent Variables", list(data.columns))
    
    



def multiple_regression():
    # Plot Graph Correlation
    st.write("**Graph Corelation Between Data :**")
    fig = sns.pairplot(data, x_vars=independent_vars, y_vars=target_var)
    st.pyplot(fig)
    # Train a Multiple Regression model and display the results
    X = data[independent_vars]
    y = data[target_var]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test, random_state=random)
    lm = LinearRegression()
    reg = lm.fit(x_train, y_train)
    y_pred = lm.predict(x_test)

    st.write("**Regression Performance Evaluation**")
    reg_score = reg.score(x_test, y_test)
    st.write("Regression Score :", reg_score )
    st.caption(f"Regression Score also mean model accuracy, your model accurancy is {reg_score*100}%")

    mse = mean_squared_error(y_test, y_pred)
    st.write("Mean Squared Error :", mse)
    mae = mean_absolute_error(y_test, y_pred)
    st.write("Mean Absolute Error :", mae)
    rmse = np.sqrt(mse)
    st.write("Root Mean Squared Error :", rmse)

    st.write("**Formula Result :**")
    st.caption("**Y = a1.x1 + a2.x2 + ... + an.xn + b**")
    coefisien = reg.coef_
    for i in range(len(coefisien)):
        st.write(f"coefisien (a{i+1})", coefisien[i])
    st.write("intercept (b) :", reg.intercept_)  
      
    # Plot Data
    st.write("**Linear Graph Actual vs Prediction**")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='g')
    ax.plot(y_pred, y_pred, color='k' )
    st.pyplot(fig)

    st.write("**Do you want to download this model ?**")
    st.caption("**WARNING THIS WILL RELOAD THE PAGE!**")
    st.download_button(
    "Download Model",
    data=pickle.dumps(lm),
    file_name="model.pkl",
    )


    predict_var = lm.predict([variable])
    st.sidebar.write("Prediction Value :",predict_var[0])
    


st.sidebar.write("_**Input Machine Learning Parameter**_")
test = st.sidebar.slider("Input test size",min_value=0.1, max_value=0.4)
random = st.sidebar.number_input("Input random state", min_value=0, max_value=50)    

ml_button = st.sidebar.button("Run")


if uploaded_file and independent_vars is not None:
    variable = []
    st.sidebar.write("_**Input Data for Prediction**_")
    for i in range(len(independent_vars)):        
        var=st.sidebar.number_input(f"Input {independent_vars[i]}")
        variable.append(var)

if ml_button:
    multiple_regression()
    

#https://github.com/sermonzagoto/machine_Larning_with_Python/blob/master/Machine_Learning_with_Python.ipynb


