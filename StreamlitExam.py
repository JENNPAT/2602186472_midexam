import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the machine learning model
model = joblib.load('trained_model.pkl')

def main():
    st.title('Mid Exam Model Deployment')

    # Min dan Max values are obtained based on the"Checking Range"
    # Numeric features
    CreditScore = st.number_input('Credit Score', min_value=0)
    Age = st.number_input('Age', min_value=18, max_value=100)
    Balance = st.number_input('Balance', min_value=0.0)
    EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0)
    NumOfProducts = st.number_input('Number of Products', min_value=1, step=1)

    # Categorical features
    Geography = st.selectbox('Geography', ['France', 'Germany','Spain'])  
    Gender = st.selectbox('Gender', ['Male', 'Female'])  
    Tenure = st.selectbox('Tenure', ['0','1','2','3','4','5','6','7','8','9','10'])  
    HasCrCard = st.selectbox('Has Credit Card?', ['Yes', 'No']) 
    IsActiveMember = st.selectbox('Is Active Member?', ['Yes', 'No'])  

    if st.button('Make Prediction'):
        # Encode categorical variables
        label_encoder = LabelEncoder()
        Geography_encoded = label_encoder.fit_transform([Geography])[0]
        Gender_encoded = label_encoder.fit_transform([Gender])[0]
        Tenure_encoded = label_encoder.fit_transform([Tenure])[0]
        HasCrCard_encoded = label_encoder.fit_transform([HasCrCard])[0]
        IsActiveMember_encoded = label_encoder.fit_transform([IsActiveMember])[0]

        features = [CreditScore, Geography_encoded, Gender_encoded, Age, Tenure_encoded, Balance, 
                    NumOfProducts, HasCrCard_encoded, IsActiveMember_encoded, EstimatedSalary]
        
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

        result = make_prediction(features)

        # transform the output label based on prediction
        if result == 0:
            output_text = "No" 
        else:
            output_text = "Yes"

        st.success(f'The prediction is: {output_text}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Reshape the input model to 2D array
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

