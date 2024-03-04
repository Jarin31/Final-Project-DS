import streamlit as st
import numpy as np

# import ml package
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

attribute_info = """
                 - Levy: $ 0- $ 5600 
                 - Manufacturer: LEXUS, CHEVROLET, HONDA, FORD, HYUNDAI, TOYOTA, MERCEDES-BENZ, OPEL, PORSCHE, BMW, JEEP, VOLKSWAGEN, AUDI, 
                                    RENAULT, NISSAN, SUBARU, DAEWOO, KIA, MITSUBISHI, SSANGYONG, MAZDA, GMC, FIAT, INFINITI, ALFA ROMEO, SUZUKI, ACURA, LINCOLN, VAZ, 
                                    GAZ,CITROEN, LAND ROVER, MINI, DODGE, CHRYSLER, JAGUAR, ISUZU, SKODA, DAIHATSU, BUICK, TESLA, CADILLAC, PEUGEOT, BENTLEY, VOLVO,
                                    სხვა, HAVAL, HUMMER, SCION, UAZ, MERCURY, ZAZ, ROVER, SEAT, LANCIA, MOSKVICH, MASERATI, FERRARI, SAAB, LAMBORGHINI, ROLLS-ROYCE, 
                                    PONTIAC, SATURN, ASTON MARTIN, GREATWALL
                 - Prod. year: 1953-2018
                 - Category: Jeep, Hatchback, Sedan, Microbus, Goods wagon, Universal, Coupe, Minivan, Cabriolet, Limousine, Pickup
                 - Leather interior: Yes, No
                 - Fuel type: Hybrid Petrol, Diesel, CNG, Plug-in Hybrid, LPG, Hydrogen
                 - Engine Volume: 0.4-20.0
                 - Mileage : 0-367000 km
                 - Cylinders : 1.0 - 12.0
                 - Drive wheels: 4x4, Front, Rear
                 - Gear box type : Automatic , Tiptronic, Variator, Manual
                 - Wheel: Left wheel, Right-hand drive
                 - Color: Silver Black White, Grey, Blue, Green, Red, Sky blue, Orange, Yellow, Brown, Golden, Beige, Carnelian red, Purple, Pink
                 - Airbags: 0-16
            
                 """

manu = {'LEXUS':1, 'CHEVROLET':2, 'HONDA':3, 'FORD':4,
       'HYUNDAI':5, 'TOYOTA':6, 'MERCEDES-BENZ':7, 'OPEL':8, 'PORSCHE':9, 'BMW':10,
       'JEEP':11,'VOLKSWAGEN':12, 'AUDI':13, 'RENAULT':14, 'NISSAN':15, 'SUBARU':16, 'DAEWOO':17, 
       'KIA':18, 'MITSUBISHI':19, 'SSANGYONG':20, 'MAZDA':21, 'GMC':22,  'FIAT':23, 'INFINITI':24, 
       'ALFA ROMEO':25, 'SUZUKI':26, 'ACURA':27, 'LINCOLN':28, 'VAZ':29, 'GAZ':30, 'CITROEN':31,
       'LAND ROVER':32, 'MINI':33, 'DODGE':34, 'CHRYSLER':35, 'JAGUAR':36, 'ISUZU':37, 'SKODA':38,
       'DAIHATSU':39, 'BUICK':40, 'TESLA':41, 'CADILLAC':42, 'PEUGEOT':43, 'BENTLEY':44, 'VOLVO':45, 'სხვა':46,
       'HAVAL':47, 'HUMMER':48, 'SCION':49, 'UAZ':50, 'MERCURY':51, 'ZAZ':52, 'ROVER':53, 'SEAT':54, 'LANCIA':55,
       'MOSKVICH':56, 'MASERATI':57, 'FERRARI':58, 'SAAB':59, 'LAMBORGHINI':60, 'ROLLS-ROYCE':61,
       'PONTIAC':62, 'SATURN':63, 'ASTON MARTIN':64, 'GREATWALL':65}
cat = {'Jeep':1, 'Hatchback':2, 'Sedan':3, 'Microbus':4, 'Goods wagon':5, 'Universal':6, 'Coupe':7, 'Minivan':8, 'Cabriolet':9, 'Limousine':10, 'Pickup':11}
leat = {'Yes':1, 'No':2}
fuel = {'Hybrid Petrol':1, 'Diesel':2, 'CNG':3, 'Plug-in Hybrid':4, 'LPG':5, 'Hydrogen':6}
gear_type = {'Automatic':1, 'Tiptronic':2, 'Variator':3, 'Manual':4}
drive_wheels = {'4x4':1, 'Front':2, 'Rear':3}
wheel =  {"Left wheel":1, 'Right-hand drive':2}
color =  {'Silver':1, 'Black White':2, 'Grey':3, 'Blue':4, 'Green':5,'Red':6, 'Sky blue':7, 
          'Orange':8, 'Yellow':9, 'Brown':10, 'Golden':11, 'Beige':12, 'Carnelian red':13, 'Purple':14, 'Pink':15}


def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value
        
def load_scaler(scaler_file):
    scaler = joblib.load(open(os.path.join(scaler_file), 'rb'))
    return scaler

def load_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model


def run_ml_app():
    st.subheader("ML Section")
    with st.expander("Attribute Info"):
        st.markdown(attribute_info)

    st.subheader("Input Your Data")
    Levy = st.number_input('Levy', 0, 5600 )
    Manufacturer = st.selectbox('Manufacturer', ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ',
                                               'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN', 'AUDI', 'RENAULT', 'NISSAN',
                                               'SUBARU', 'DAEWOO' ,'KIA', 'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT',
                                               'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'ACURA' ,'LINCOLN', 'VAZ', 'GAZ', 'CITROEN',
                                                'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR', 'ISUZU', 'SKODA', 'DAIHATSU',
                                                'BUICK', 'TESLA' ,'CADILLAC' ,'PEUGEOT', 'BENTLEY','VOLVO', 'სხვა', 'HAVAL' ,'HUMMER',
                                                'SCION', 'UAZ', 'MERCURY' ,'ZAZ', 'ROVER' 'SEAT', 'LANCIA', 'MOSKVICH' ,'MASERATI', 
                                                'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE', 'PONTIAC' ,'SATURN', 'ASTON MARTIN' ,'GREATWALL'])
    Prod_year = st.number_input('Product Year', 1952, 2018)
    Category = st.selectbox('Category', ['Jeep', 'Hatchback' ,'Sedan', 'Microbus', 'Goods wagon' ,'Universal' ,'Coupe','Minivan' ,'Cabriolet', 'Limousine','Pickup'])
    Leather_interior = st.radio('Leather interior', ['Yes', 'No'])
    Fuel_type = st.selectbox('Fuel type', ['Hybrid', 'Petrol', 'Diesel' ,'CNG', 'Plug-in Hybrid' ,'LPG' ,'Hydrogen'])
    Engine_volume = st.number_input("Engine volume", 0, 20)
    Mileage = st.number_input("Mileage", 0, 367000)
    Cylinders = st.number_input("Cylinders",10,60)
    Gearbox_type = st.selectbox("Gear box type",['Automatic', 'Tiptronic' ,'Variator', 'Manual'])
    Drive_wheels = st.selectbox("Drive wheels:",['4x4' ,'Front', 'Rear'])
    Wheel = st.radio("Wheel", ['Left wheel', 'Right-hand drive'])
    Color = st.selectbox("Color",['Silver','Black', 'White' ,'Grey' ,'Blue' ,'Green' ,'Red' ,'Sky blue', 'Orange','Yellow', 'Brown','Golden', 'Beige', 'Carnelian red' ,'Purple' ,'Pink'])
    Airbags = st.number_input('Airbags', 0,16)

    with st.expander("Your Selected Options"):
        result = {
            'Levy': Levy,
            'Manufacturer':Manufacturer,
            'Product Year': Prod_year,
            'Category':Category,
            'Leather interior':Leather_interior,
            'Fuel type': Fuel_type,
            'Engine volume':Engine_volume,
            'Mileage':Mileage,
            'Cylinders':Cylinders,
            'Gearbox type':Gearbox_type,
            'Drive wheels:':Drive_wheels,
            'Wheel':Wheel,
            'Color':Color,
            'Airbags':Airbags,
        }
    
    # st.write(result)

    encoded_result = []
    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        elif i in ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ',
                    'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN', 'AUDI', 'RENAULT', 'NISSAN',
                    'SUBARU', 'DAEWOO' ,'KIA', 'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT',
                    'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'ACURA' ,'LINCOLN', 'VAZ', 'GAZ', 'CITROEN',
                    'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR', 'ISUZU', 'SKODA', 'DAIHATSU',
                    'BUICK', 'TESLA' ,'CADILLAC' ,'PEUGEOT', 'BENTLEY','VOLVO', 'სხვა', 'HAVAL' ,
                    'HUMMER', 'SCION', 'UAZ', 'MERCURY' ,'ZAZ', 'ROVER' 'SEAT', 'LANCIA', 'MOSKVICH' ,'MASERATI', 
                    'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE', 'PONTIAC' ,'SATURN', 'ASTON MARTIN' ,'GREATWALL']:
            res = get_value(i, manu)
            encoded_result.append(res)
        elif i in ['Jeep', 'Hatchback' ,'Sedan', 'Microbus', 'Goods wagon' ,'Universal' ,
                   'Coupe','Minivan' ,'Cabriolet', 'Limousine','Pickup']:
            res = get_value(i, cat)
            encoded_result.append(res)
        elif i in ['Yes', 'No']:
            res = get_value(i, leat)
            encoded_result.append(res)
        elif i in ['Hybrid', 'Petrol' ,'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen']:
            res = get_value(i, fuel)
            encoded_result.append(res)
        elif i in ['Automatic' ,'Tiptronic', 'Variator' ,'Manual']:
            res = get_value(i, gear_type)
            encoded_result.append(res)
        elif i in ['4x4', 'Front', 'Rear']:
            res = get_value(i, drive_wheels)
            encoded_result.append(res)
        elif i in ['Left wheel' ,'Right-hand drive']:
            res = get_value(i, wheel)
            encoded_result.append(res)
        elif i in ['Silver','Black' ,'White', 'Grey','Blue', 'Green' ,'Red' ,'Sky blue', 'Orange', 'Yellow', 'Brown', 'Golden', 'Beige', 'Carnelian red' ,'Purple', 'Pink']:
            res = get_value(i, color)
            encoded_result.append(res)

    
    # Load hasil scaling
    scaler = load_scaler('scaled_data.pkl')

    # Misalnya, single_array adalah data yang ingin Anda skalakan
    single_array = np.array(encoded_result).reshape(1, -1)

    # Scaling data
    scaled_data2 = scaler.transform(single_array)

    
    #prediction section
    st.subheader('Prediction Result')

    model = load_model("model_final_pro.pkl")

    # Lakukan prediksi
    prediction = np.expm1(model.predict(scaled_data2))

    # Menampilkan hasil prediksi
    st.write(prediction)

