import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math




best_model = pickle.load(open('best_model.pkl','rb'))

brand_names = pickle.load(open('brand_name.pkl','rb'))
model_names = pickle.load(open('model_name.pkl','rb'))

cleaned_data = pd.read_csv('cleaned_car_data.csv')

st.title('Car Price Prediction')
# st.sidebar.title('Enter Car Details')

col1, col2 = st.columns(2)

# Input fields for the first column
with col1:
    year = st.slider('Year', min_value= cleaned_data['year'].min(), max_value= cleaned_data['year'].max())
    km_driven = st.slider('Kilometers Driven', min_value=cleaned_data['km_driven'].min(), max_value=cleaned_data['km_driven'].max())
    fuel = st.selectbox('Fuel Type', cleaned_data['fuel'].unique())
    seller_type = st.selectbox('Seller Type', cleaned_data['seller_type'].unique())

# Input fields for the second column
with col2:
    transmission = st.selectbox('Transmission', cleaned_data['transmission'].unique())
    owner = st.selectbox('Owner', cleaned_data['owner'].unique())
    brand = st.selectbox('Brand', cleaned_data['brand'].unique())
    brand_filter = cleaned_data[cleaned_data['brand'] == brand]
    model = st.selectbox('Model', brand_filter['model'].unique())




input_data = pd.DataFrame({
    'year' : [year],
    'km_driven' : [km_driven],
    'fuel' : [fuel],
    'seller_type' :[seller_type],
    'transmission' : [transmission],
    'owner' : [owner],
    'brand' : [brand],
    'model' : [model]

})

encoder = LabelEncoder()
fuel_encoded = encoder.fit_transform([fuel])[0]
seller_type_encoded = encoder.fit_transform([seller_type])[0]
transmission_encoded = encoder.fit_transform([transmission])[0]
owner_encoded = encoder.fit_transform([owner])[0]
brand_encoded = encoder.fit_transform([brand])[0]
model_encoded = encoder.fit_transform([model])[0]
    # Create DataFrame from user inputs
input_data = pd.DataFrame({
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel_encoded],
        'seller_type': [seller_type_encoded],
        'transmission': [transmission_encoded],
        'owner': [owner_encoded],
        'brand' : [brand_encoded],
        'model' : [model_encoded]
    })

if st.button('prediction'):
   
    prediction = best_model.predict(input_data)
    # predict_price = math.ceil(prediction[0]/1000)
    st.success(f'Predicted Selling Price: {prediction[0]}  INR')

