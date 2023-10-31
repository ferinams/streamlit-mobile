import pickle
import streamlit as st

model = pickle.load(open('mobile_phone.sav', 'rb'))

st.title('Estimasi Harga Telepon Genggam')

Battery_capacity_mAh = st.number_input('input kapasitas baterai')
Screen_size_inches = st.number_input('input ukuran layar')
Processor = st.number_input('input Processor')
RAM_MB = st.number_input('input besarnya RAM')
Internal_storage_GB = st.number_input('input besarnya memori internal')
Rear_camera = st.number_input('input pixel kamera belakang')                                  
Number_of_SIMs = st.number_input('input banyaknya sim card')
Resolution_x   = st.number_input('inputkan resolusi layar')                        
                              
predict = ''

if st.button('cek harga') : 
    predict = model.predict(
        [[Battery_capacity_mAh, Screen_size_inches, Processor, RAM_MB, Internal_storage_GB, Rear_camera, Number_of_SIMs]]
    )

    st.write ('estmasi harga saham dalam IDR (juta)', predict)