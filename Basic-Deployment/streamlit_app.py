import pickle
import streamlit as st

LATITUDE = 40.723424241212
LONGITUDE = -72.2342124535

st.title('Dự đoán giá nhà ở Niu Yóc')
st.write('Ứng dụng dự đoán giá nhà dựa vào diện tích và số lượng phòng ngủ.')

@st.cache_resource
def load_model():
    with open('/home/rowan/Rowan/MLops/Basic-Deployment/notebook/rf_regressor.pkl', 'rb') as file:
        return pickle.load(file)
    
model = load_model()

def predict_price(bedrooms, bathrooms, size):
    x = [[bedrooms, bathrooms, size, LATITUDE, LONGITUDE]]
    a = model.predict(x)
    return a

bedrooms = st.number_input('Phong ngu', min_value=1, max_value=12, value=1, step=1)
bathrooms = st.number_input('Phong tam', min_value=1, max_value=11, value=1, step=1)
size = st.number_input('Dien tich', min_value=111, max_value=11111, value=111, step=10)

if st.button('du doan'):
    price = predict_price(bedrooms, bathrooms, size)[0]
    st.write(f'The estimated price of this house is ${price:,.2f}')