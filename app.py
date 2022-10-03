import pandas as pd
from prophet import Prophet
import streamlit as st
from PIL import Image

st.title('Amazon Stock Price Predictor')
st.write('Made using *[Fbprophet](https://facebook.github.io/prophet/)!*:sunglasses:')
df=pd.read_csv('Amazon.csv')
df['Date']=pd.to_datetime(df['Date'])
df.drop(['Open','High','Low','Adj Close','Volume'],axis=1,inplace=True)

df=df.rename(columns={'Date':'ds','Close':'y'})

col1, col2, col3 = st.columns(3)
with col1:
    y=str(st.selectbox('Year',(range(2020,2031))))
with col2:
    m=str(st.selectbox('Month',(range(1,13))))
    
with col3:
    d=str(st.selectbox('Day',(range(1,32))))


if len(m)<2:
    m="0"+m
if len(d)<2:
    d='0'+d

date=y+'-'+m+'-'+d

model=Prophet()
model.fit(df)
future=model.make_future_dataframe(periods=3650)
predict=model.predict(future)

result=predict[predict['ds']==date]['yhat'].values[0]

idx=predict.index[predict['ds']==date][0]


if st.button('Predict'):
    st.write('The stock price will be',round(result,2),'on date',date)
    st.header('Predictions of last 500 days and next 500 days')
    st.pyplot(model.plot(predict.iloc[-500+idx:idx+500,:],include_legend=True,figsize=(15,6)))

    st.header('Trends and Seasonality')
    st.pyplot(model.plot_components(predict.iloc[-500+idx:idx+500,:]))

    





