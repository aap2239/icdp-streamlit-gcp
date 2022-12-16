import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
from io import StringIO
import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import plotly

st.markdown("<h1 style='text-align: center; color: red;'>AMEX Credit Default Prediction</h1>", unsafe_allow_html=True)


project_main= st.sidebar.selectbox('What would you like to see?', ('EDA','Shapley','Model_prediction'))


if project_main=='EDA':
    EDA = st.sidebar.selectbox('Select Charts/Plot type', ('Customer Information', 'Feature Distributions', 'Correlation Plots'))


    if EDA == 'Customer Information':
        selected_status = st.sidebar.selectbox('Which Plots',
                                        options = ['Feature Type', 'Last Statements', 'Statements per customer', 
                                                    'Default Distribution By Day','Target Distribution'])

        if selected_status == 'Feature Type':
            image = plotly.io.read_json("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\Featuretype.json")
            st.plotly_chart(image)

        if selected_status == 'Last Statements':
            image = plotly.io.read_json("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\Laststatements.json")
            st.plotly_chart(image, caption='Last Statements')

        if selected_status == 'Statements per customer':
            image = plotly.io.read_json("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\CustomerstatementFreq.json")
            st.plotly_chart(image, caption='Statements per customer')

        if selected_status == 'Default Distribution By Day':
            image = plotly.io.read_json("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\Defaultbyday.json")
            st.plotly_chart(image, caption='Default by day')

        if selected_status == 'Target Distribution':
            image = plotly.io.read_json("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\Targetdistr.json")
            st.plotly_chart(image)
            
        

    elif EDA == 'Feature Distributions':
        selected_status = st.sidebar.selectbox('Which features',
                                        options = ['Payment', 'Balance', 
                                                    'Spend', 'Risk','Categorical'])
        if selected_status == 'Payment':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\payment distr.png")
            st.image(image, caption='Payment Distribution')

        if selected_status == 'Balance':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\balance distr.png")
            st.image(image, caption='Balance Distribution')

        if selected_status == 'Spend':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\spend distr.png")
            st.image(image, caption='Spend Distribution')

        if selected_status == 'Risk':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\risk distr.png")
            st.image(image, caption='Risk Distribution')

        if selected_status == 'Categorical':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\Categorical distr.png")
            st.image(image, caption='Categorical Distribution')
    
    elif EDA == 'Correlation Plots':
        selected_status = st.sidebar.selectbox('Which features',
                                        options = ['Payment', 'Balance', 
                                                    'Spend', 'Risk'])
        if selected_status == 'Payment':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\payment corr.png")
            st.image(image, caption='Payment Correlation')

        if selected_status == 'Balance':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\balance corr.png")
            st.image(image, caption='Balance Correlation')

        if selected_status == 'Spend':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\spend corr.png")
            st.image(image, caption='Spend Correlation')

        if selected_status == 'Risk':
            image = Image.open("C:\\Users\\mihee\\OneDrive\\Documents\\Project\\risk corr.png")
            st.image(image, caption='Risk Correlation')



elif project_main=='Shapley':
    print('Shapley')


elif project_main=='Model_prediction':
    # loading the trained model
    pickle_in = open('C:\\Users\\mihee\\Downloads\\classifier.pkl', 'rb') 
    classifier = pickle.load(pickle_in)

    uploaded_file = st.file_uploader(" ")

    if uploaded_file is not None:
        test = spark.read.option('header','true').csv(uploaded_file) ##or pd.read_csv(uploaded_file)
        st.write(test)

    
    @st.cache()
    
    # defining the function which will make the prediction using the data which the user inputs 
    def prediction(inputs):   
    
        # Making predictions 
        prediction = classifier.predict(inputs)
        
        if prediction == 0:
            pred = 'Not Defaulted'
        else:
            pred = 'Defaulted'
        return pred
        
    
    # this is the main function in which we define our webpage  
    def main():       
        # front end elements of the web page 
        html_temp = """ 
        <div style ="background-color:yellow;padding:13px"> 
        <h1 style ="color:black;text-align:center;">Default Prediction</h1> 
        </div> 
        """
        
        # display the front end aspect
        st.markdown(html_temp, unsafe_allow_html = True) 
        
        # following lines create boxes in which user can enter data required to make prediction 
        
        result =""
        
        # when 'Predict' is clicked, make the prediction and store it 
        if st.button("Predict"): 
            result = prediction(test) 
            st.success('This customer has {}'.format(result))
        
    if __name__=='__main__': 
        main()



