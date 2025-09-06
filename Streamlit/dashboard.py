import streamlit as st
import plotly.express as ox
import pandas as pd 
import os 
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(page_title='Chasfat Projects!!!',page_icon=':bar_chart',layout='wide')
st.title(' :bar_chart: Sample Superstore EDA')
st.markdown('<style> div.block-container{padding-top:1rem;} </style>',unsafe_allow_html=True)

f1=st.file_uploader(':file_folder: Upload a file',type=(['csv','txt','xlsx','xls']))

if f1 is not None:
    filename=f1.name
    st.write(filename)
    df=pd.read_csv(filename)
else:
    os.chdir(r'C:/Users/user/Desktop/DataScience')
    df=pd.read_csv('C:/Users/user/Desktop/DataScience/Data/Sample - Superstore.xls',encoding='ISO-8859-1')

col1,col2=st.columns((2))
df['Order Date']=pd.to_datetime(df['Order Date'])

#Getting the min and max date 
startDate = pd.to_datetime(df['Order Date']).min()
endDate= pd.to_datetime(df['Order Date']).max()

with col1:
    date1=pd.to_datetime(st.date_input('Start Date',startDate))

with col2:
    date2=pd.to_datetime(st.date_input('End Date',endDate))

df = df[(df['Order Date'] >= date1 ) & (df['Order Date'] <= date2 )].copy