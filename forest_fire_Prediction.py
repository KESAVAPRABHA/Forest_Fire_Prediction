import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
import streamlit as st

data = pd.read_csv('Downloads/forestfires.csv')  #Ensure your path

encoder_month = LabelEncoder()
encoder_day = LabelEncoder()
data['month'] = encoder_month.fit_transform(data['month'])
data['day'] = encoder_day.fit_transform(data['day'])


X = data.drop(columns=['area'])
y = data['area']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Model Mean Squared Error: {mse}")


joblib.dump(model, 'forest_fire_model.pkl')


model = joblib.load('forest_fire_model.pkl')


st.title("Forest Fire Impact Prediction")
st.write("Enter values for the following parameters to predict the burned area:")

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']


month = st.selectbox("Month", months)
day = st.selectbox("Day of the week", days)
temp = st.slider("Temperature (°C)", 0.0, 50.0, step=0.1)
RH = st.slider("Relative Humidity (%)", 0, 100, step=1)
wind = st.slider("Wind Speed (km/h)", 0.0, 10.0, step=0.1)
rain = st.slider("Rainfall (mm)", 0.0, 10.0, step=0.1)
FFMC = st.slider("FFMC index", 0.0, 100.0, step=0.1)
DMC = st.slider("DMC index", 0.0, 300.0, step=0.1)
DC = st.slider("DC index", 0.0, 800.0, step=0.1)
ISI = st.slider("ISI index", 0.0, 50.0, step=0.1)


month_encoded = encoder_month.transform([month])[0]
day_encoded = encoder_day.transform([day])[0]


if st.button("Predict Burned Area"):
    input_data = np.array([[month_encoded, day_encoded, FFMC, DMC, DC, ISI, temp, RH, wind, rain]])
    
    input_data = np.append(input_data, [0, 0]) 
    input_data = input_data.reshape(1, -1) 

    prediction = model.predict(input_data)
    st.write(f"Predicted Burned Area: {prediction[0]:.2f} hectares")