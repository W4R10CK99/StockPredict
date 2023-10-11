import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st

st.title('Stock Trend Predictor')


start = '2010-01-01'
end = '2022-12-31'

user_input = st.text_input('Enter Stock Ticker','AAPL')

df = yf.download(user_input,start=start,end=end)

#describing data

st.subheader('Data from 2010 - 2022')
st.write(df.describe())

#visualisation

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 MA & 200 MA')
MA100 = df.Close.rolling(100).mean()
MA200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close,'b',label='Actual')
plt.plot(MA100,'g', label='100 Moving Average')
plt.plot(MA200,'r', label='200 Moving Average')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

#load the trained model
model = load_model('keras_model.h5')


past_100_days = data_train.tail(100)
final_df = pd.concat([past_100_days, data_test], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])


x_test = np.array(x_test)
y_test = np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]

y_test = y_test * scale_factor
y_predicted = y_predicted * scale_factor


st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.legend()
st.pyplot(fig2)
