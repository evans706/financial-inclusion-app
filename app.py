import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('financial_inclusion_model.pkl')

def main():
    st.title("Financial Inclusion in Africa Predictor")
    st.write("Predict whether an individual is likely to have a bank account.")

    # Create the form
    with st.form("prediction_form"):
        st.header("Enter Customer Details")

        # Define the input fields matching the training data features
        # Mappings based on the LabelEncoder steps from the notebook
        
        country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
        location_type = st.selectbox("Location Type", ["Rural", "Urban"])
        cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
        household_size = st.number_input("Household Size", min_value=1, max_value=20, value=4)
        age_of_respondent = st.number_input("Age", min_value=16, max_value=100, value=25)
        gender_of_respondent = st.selectbox("Gender", ["Female", "Male"])
        
        relationship_with_head = st.selectbox("Relationship with Head", 
                                              ["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"])
        
        marital_status = st.selectbox("Marital Status", 
                                      ["Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated", "Dont know"])
        
        education_level = st.selectbox("Education Level", 
                                       ["Primary education", "No formal education", "Secondary education", "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"])
        
        job_type = st.selectbox("Job Type", 
                                ["Self employed", "Government Dependent", "Formally employed Private", "Informal dependent", "Formally employed Government", "Farming and Fishing", "Remittance Dependent", "Other Income", "Dont Know/Refuse to answer", "No Income"])

        # Submit button
        submit_val = st.form_submit_button("Predict Financial Inclusion")

    if submit_val:
        # 1. Preprocess the input data (Convert text to numbers manually as per LabelEncoder)
        # Note: In a production app, you would load the encoders. For this checkpoint, simple mapping is sufficient.
        
        # Mappings (These must match exactly what your LabelEncoder output in Phase 1)
        # Example mappings below (You might need to adjust based on your specific encoding run):
        
        def encode_inputs(val, options):
            return options.index(val) # Simple index mapping if LabelEncoder sorted alphabetically

        # Manual Data Preparation
        # We need to create a DataFrame with columns in the EXACT same order as X_train
        
        # Mappings logic (simplified for demonstration)
        # IMPORTANT: Replace these dictionaries with the actual logic if your LabelEncoder didn't sort alphabetically
        country_map = {"Kenya": 0, "Rwanda": 1, "Tanzania": 2, "Uganda": 3}
        loc_map = {"Rural": 0, "Urban": 1}
        cell_map = {"No": 0, "Yes": 1}
        gender_map = {"Female": 0, "Male": 1}
        
        # For complex categories, usually LabelEncoder sorts alphabetically. 
        # Ideally, load the 'encoders' dict from Phase 1, but we will rely on sorted lists here.
        rel_opts = sorted(["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"])
        mar_opts = sorted(["Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated", "Dont know"])
        edu_opts = sorted(["Primary education", "No formal education", "Secondary education", "Tertiary education", "Vocational/Specialised training", "Other/Dont know/RTA"])
        job_opts = sorted(["Self employed", "Government Dependent", "Formally employed Private", "Informal dependent", "Formally employed Government", "Farming and Fishing", "Remittance Dependent", "Other Income", "Dont Know/Refuse to answer", "No Income"])

        # Construct vector
        row = [
            country_map[country],
            loc_map[location_type],
            cell_map[cellphone_access],
            household_size,
            age_of_respondent,
            gender_map[gender_of_respondent],
            rel_opts.index(relationship_with_head),
            mar_opts.index(marital_status),
            edu_opts.index(education_level),
            job_opts.index(job_type)
        ]

        # Convert to DataFrame (ensure columns match X.columns from training)
        cols = ['country', 'location_type', 'cellphone_access', 'household_size', 
                'age_of_respondent', 'gender_of_respondent', 'relationship_with_head', 
                'marital_status', 'education_level', 'job_type']
        
        X_new = pd.DataFrame([row], columns=cols)

        # Make Prediction
        prediction = model.predict(X_new)[0]
        probability = model.predict_proba(X_new)[0][1]

        # Display Result
        if prediction == 1:
            st.success(f"Prediction: This individual is **likely** to have a bank account. (Confidence: {probability:.2f})")
        else:
            st.warning(f"Prediction: This individual is **unlikely** to have a bank account. (Confidence: {1-probability:.2f})")

if __name__ == '__main__':
    main()