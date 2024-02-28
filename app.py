import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder



best_model = pickle.load(open('best_model.pkl','rb'))

brand_names = pickle.load(open('brand_name.pkl','rb'))
model_names = pickle.load(open('model_name.pkl','rb'))

st.title('Car Price Prediction')
st.sidebar.title('Enter Car Details')
year = st.sidebar.number_input('year', min_value= 2000, max_value=2022, step=1)
km_driven = st.sidebar.number_input('Kilometers Driven', min_value=0)
fuel = st.sidebar.selectbox('Fuel Type', {'Petrol': 1, 'Diesel': 2, 'CNG':3})
seller_type = st.sidebar.selectbox('Seller Type', ['Dealer', 'Individual'])
transmission = st.sidebar.selectbox('Transmission', ['Manual', 'Automatic'])
owner = st.sidebar.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
brand = st.sidebar.selectbox('Brand', brand_names)
model = st.sidebar.selectbox('Model', model_names)

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

if st.sidebar.button('prediction'):
   
    prediction = best_model.predict(input_data)
    predict_price = math.ceil(prediction[0]/1000)
    st.success(f'Predicted Selling Price: {predict_price} lakh INR')

