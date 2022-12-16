import pandas as pd 
import numpy as np

import streamlit as st 
import os

from PIL import Image
import pickle
import pandas as pd
import plotly
from io import StringIO

from helper_funcs import *

def main():

    st.title('Explainable Credit Default Prediction')

    PAGES = {
        'Home': homepage,
        'Exploratory Data Analysis': eda,
        'Upload and Classify': upload,
        'Shapley Explanation': shap_explain,
    }
    st.sidebar.title('Navigation')
    PAGES[st.sidebar.radio('Go To', ('Home', 'Exploratory Data Analysis', 'Upload and Classify', 'Shapley Explanation'))]()
    # st.write(option_chosen)
    
def homepage():
    st.write("""
    ## Explainable Credit Default Prediction
    """)

    st.markdown("<p>Credit card default prediction has become a crucial key component of the financial industry. The industry has become heavily reliant on the default prediction mechanism in order to function and manage customers and profits in an optimized manner. In the past few years, this issue of credit card default has had an adverse impact on the economy and financial institutions, ultimately affecting consumers. This issue arises when an individual fails to comply and pay back the dues in a given required period of time which leads to losses to the financial institutions.</p>"

"<p>So, in order to deal with this problem the industry has evolved and leveraged consumer data to apply statistical methods and predict the default behavior of the consumers in order to prevent sanctioning of further loans and credit to such consumers and setting up policies for the same. But these advancements are based on the incorporation of huge data and predictions based on statistical methods that are highly dependent on users' interpretation and judgment which is prone to human error and bias. These methods have been found to be inefficient and not to be optimized in a manner to be deployed on a large scale and be heavily relied upon for dealing with this issue. Moreover, these traditional statistical methods are unable to leverage the possible potential of big data and fail to provide significant accuracy and efficiency for making decisions.</p>"

"<p>Recently, with the advancement in the field of big data and machine learning, the industry has been able to identify the need to integrate big data techniques to achieve the required solution to these issues and have a significantly better model than previous solutions for credit card default prediction. With the use of machine learning techniques, credit card default can be predicted using customers' past data associated with the financial institution based on different factors and variables that affect or lead to such defaults. Big data and machine learning algorithms are able to provide logical and optimized solutions with scope for improvement and continuous growth to achieve significant outcomes and profitability in the financial industry.</p>" 

"<p>In this project, we utilize the real potential of big data by leveraging the large-scale AMEX dataset which helps us define a realistic model matching industrial standards and thus, making our work more relevant and realistic to be adopted by the industry. We build an end-to-end cloud model based on the Google Cloud Platform and use apache spark engines for each and every step of our model. The pipeline of the model includes data sourcing from cloud storage, data preprocessing, feature engineering, exploratory data analysis, model training, and finally, deployment of the model on the front end. This work specifically makes use of spark engines to get better and more significant efficiency in dealing with such large volumes of data in the best possible optimized way to make sure that the least memory and time is utilized and can be deployed easily on a front-end platform. The work also focuses to provide a solution for explainability by employing game theory techniques like shapley\cite{rw-9}. Incorporating explainability makes our model more understandable and easy to interpret, thus making the model more adaptable and usable by the industry as it is easier to reason out the behavior of the model.</p>"
                , unsafe_allow_html=True)

def eda():
    st.write("""
    ## Exploratory Data Visualization
    """)
    chart_visual = st.sidebar.selectbox('Select Charts/Plot type', 
                                    ('Customer Information', 'Feature Distributions', 'Correlation Plots'))
    if chart_visual == 'Customer Information':
        selected_status = st.sidebar.selectbox('Which Plots',
                                           options = ['Feature Type', 'Last Statements', 'Statements per customer', 
                                                      'Default Distribution By Day','Target Distribution'])

        if selected_status == 'Feature Type':
            image = plotly.io.read_json("/home/aap2239/icdp-streamlit-gcp/assets/plots/Featuretype.json")
            st.plotly_chart(image)

        if selected_status == 'Last Statements':
            image = plotly.io.read_json("/home/aap2239/icdp-streamlit-gcp/assets/plots/Laststatements.json")
            st.plotly_chart(image, caption='Last Statements')

        if selected_status == 'Statements per customer':
            image = plotly.io.read_json("/home/aap2239/icdp-streamlit-gcp/assets/plots/CustomerstatementFreq.json")
            st.plotly_chart(image, caption='Statements per customer')

        if selected_status == 'Default Distribution By Day':
            image = plotly.io.read_json("/home/aap2239/icdp-streamlit-gcp/assets/plots/Defaultbyday.json")
            st.plotly_chart(image, caption='Default by day')

        if selected_status == 'Target Distribution':
            image = plotly.io.read_json("/home/aap2239/icdp-streamlit-gcp/assets/plots/Targetdistr.json")
            st.plotly_chart(image)
            
    elif chart_visual == 'Feature Distributions':
        selected_status = st.sidebar.selectbox('Which features',
                                           options = ['Payment', 'Balance', 
                                                      'Spend', 'Risk'])
        if selected_status == 'Payment':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/payment distr.png")
            st.image(image, caption='Payment Distribution')

        if selected_status == 'Balance':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/balance distr.png")
            st.image(image, caption='Balance Distribution')

        if selected_status == 'Spend':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/spend distr.png")
            st.image(image, caption='Spend Distribution')

        if selected_status == 'Risk':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/risk distr.png")
            st.image(image, caption='Risk Distribution')
  
    elif chart_visual == 'Correlation Plots':
        selected_status = st.sidebar.selectbox('Which features',
                                           options = ['Payment', 'Balance', 
                                                      'Spend', 'Risk'])
        if selected_status == 'Payment':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/payment corr.png")
            st.image(image, caption='Payment Correlation')

        if selected_status == 'Balance':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/balance corr.png")
            st.image(image, caption='Balance Correlation')

        if selected_status == 'Spend':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/spend corr.png")
            st.image(image, caption='Spend Correlation')

        if selected_status == 'Risk':
            image = Image.open("/home/aap2239/icdp-streamlit-gcp/assets/plots/Risk corr.png")
            st.image(image, caption='Risk Correlation') 

def upload():
    st.write("""
    ## Predictor
    """)
    dataset_path = st.text_input('Dataset Path', 'Enter Dataset Path here!') 
    limit_of_df = st.number_input("Limit of Rows (No limit if it is set to -1)", value = -1, min_value = -1, max_value = None, step = 100)
    if st.button("Get Predictions"):
        with st.spinner("Predicting"):
            if limit_of_df == -1:
                limit_of_df = None
            preds = pipeline_pred(
                path = dataset_path,
                limit_of_df = limit_of_df,
                spark = spark,
                models_dict = models_dict,
                meta_data = meta_data,
            )
            preds = preds.limit(50).toPandas()
            st.dataframe(preds)

def shap_explain():
    pass


spark = get_spark_session()
meta_data = load_meta_data()
models_dict = load_models(spark = spark)
if __name__ == "__main__":
    main()