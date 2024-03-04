import streamlit as st
import streamlit.components.v1 as stc

from ml_app import run_ml_app

html_temp = """
            <div style="background-color:#3872fb;padding:10px;border-radius:10px">
		    <h1 style="color:white;text-align:center;">Price Prediction App </h1>
		    <h4 style="color:white;text-align:center;">2DD Team </h4>
		    </div>
            """

desc_temp = """
            ### Price Prediction App
            This app will be used by the 2DD team to predict whether the employee get a promotion or not
            #### Data Source
            - https://raw.githubusercontent.com/Jarin31/Final-Project-DS/main/car_price_prediction.csv
            #### App Content
            - Exploratory Data Analysis
            - Machine Learning Section
            """

def main():

    stc.html(html_temp)
    
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader("Welcome to Homepage")
        st.markdown(desc_temp)
    elif choice == "Machine Learning":
        # st.subheader("Welcome to Machine learning")
        run_ml_app()


if __name__ == '__main__':
    main()
