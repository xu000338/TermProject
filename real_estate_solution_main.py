'''
CST 2216
Term Project
David Xu (041173885)
real_estate_solution_main.py
Description: Front-end code for real_estate_solution_model
'''


import src.real_estate_solution_model as real_estate_solution_model
import pandas as pd
import streamlit as st
import src.logger as logger

@logger.exception_decorator
def main():
    
    #begin streamlit code
    st.set_page_config(
            page_title="Real Estate Price Prediction App",
        )
    st.title("Real Estate Price Prediction")
    st.header("""
        This app uses decision tree or random forest models to predict real estate prices based on user input.
        """)


    year_sold = st.text_input("Year")
    
    property_tax = st.number_input("Property Tax")
    
    insurance = st.number_input("Insurance")
    
    beds = st.number_input("No. of Beds")
    
    baths = st.number_input("No. of Bathrooms")
    
    sqft = st.number_input("Are in Square Feet")
    
    year_built = st.text_input("Year Built")
    
    lot_size = st.number_input("Lot Size")
    
    basement = st.selectbox("Basement", [0, 1])
    
    popular = st.selectbox("Popular: ", [0, 1])
    
    recession = st.selectbox("Recession ", [0, 1])
    
    property_age = st.number_input("Property Age")
    
    property_type = st.radio("Property Type: ", ('Bunglow', 'Condo'))
    
    if(property_type == 'Bunglow'):
        property_type_Bunglow = 1
        property_type_Condo = 0
    else:
        property_type_Condo = 1
        property_type_Condo = 0
        
    methodology = st.radio("Select methodology: ", ("Decision Tree", "Random Forest"))

    if methodology == "Decision Tree":
        st.success("Predicting property price with decision tree!")
    else:
        st.success("Predicting property price with random forest!")
        

    if(st.button("Predict Price")):
        if methodology == "Decision Tree":
            dt_price =  real_estate_solution_model.DecisionTreeApp(year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular, recession, property_age, property_type_Bunglow, property_type_Condo)
            st.write(dt_price)         
        else:
            rf_price =  real_estate_solution_model.RandomForestApp(year_sold, property_tax, insurance, beds, baths, sqft, year_built, lot_size, basement, popular, recession, property_age, property_type_Bunglow, property_type_Condo)
            st.write(rf_price)      
          
    
    
if __name__ == "__main__":
    main()