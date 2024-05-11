import yfinance #Also install it 
import streamlit as st 
import pandas as pd 

st.write(""" 

#Simple Stock Price App

Shown are the stock closing price and volume of Google!

         
         """)
tickerSymbol='GOOGL'
#Get data on this ticker 
tickerData=yf.Ticker(tickerSymbol)
#Get the historical prices of this ticker 
tickerDf=tickerData.history(period='id',start='2010-5-31',end='2020-5-31')
#Open high low close volume dividends stock splits 


st.line_chart(tickerDf.close)
st.line_chart(tickerDf.volume)








v

