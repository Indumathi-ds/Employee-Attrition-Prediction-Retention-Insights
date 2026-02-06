import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, accuracy_score
import matplotlib.pyplot as plt


st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title('Employee Attrition Predictor')

st.subheader('Upload CSV File')
uploaded_file = st.file_uploader("Upload HR CSV file", type="csv")

if uploaded_file is not None:
    data=pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview:")
    st.write(data.head(5))
    
    data.dropna(axis=0, inplace=True)  # Drop rows with missing values
    data.drop_duplicates(inplace=True)  
    
    data["Attrition"] = data["Attrition"].map({"Yes": 1, "No": 0}).astype(int)
    
    features = [
        "Age", "MonthlyIncome", "OverTime", "JobRole", 
        "Department", "JobLevel", "YearsAtCompany", 
        "Education", "MaritalStatus", "Gender"
    ]

    x = data[features]
    x = pd.get_dummies(x, drop_first=True).astype(int)
    y = data["Attrition"]
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ))
    ])


    classifier.fit(x_train, y_train)
    y_probs = classifier.predict_proba(x_test)[:,1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    gmeans = (tpr * (1 - fpr))**0.5
    best_idx = gmeans.argmax()
    best_threshold = thresholds[best_idx]
    
    # Accuracy with default 0.5
    default_acc = accuracy_score(y_test, (y_probs >= 0.5).astype(int))
    st.success(f"Model Accuracy (0.5 threshold): {default_acc:.2f}")
    st.info(f"Suggested Threshold based on ROC curve: {best_threshold:.2f}")
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC Curve')
    ax.scatter(fpr[best_idx], tpr[best_idx], color='red', label=f'Best Threshold = {best_threshold:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)
    st.subheader('Predict for a Single Employee')

    # Input field 
    age= st.number_input("Age",18,60)
    monthly_income=st.number_input("MonthlyIncome:",min_value=1000)
    overtime=st.selectbox("OverTime",["Yes","No"])
    jobrole = st.selectbox("Job Role", data["JobRole"].unique())
    department = st.selectbox("Department", data["Department"].unique())
    joblevel = st.number_input("Job Level", 1, 5)
    years_at_company = st.number_input("Years at Company", 0, 40)
    education = st.selectbox("Education Level", sorted(data["Education"].unique()))
    marital_status = st.selectbox("Marital Status", data["MaritalStatus"].unique())
    gender = st.selectbox("Gender", data["Gender"].unique())

   
    if st.button("Predict Attrition"):
        input_df=pd.DataFrame({
            "Age":[age],
            "MonthlyIncome":[monthly_income],
            "OverTime": [overtime],
            "JobRole":[jobrole],
            "Department":[department],
            "JobLevel":[joblevel],
            "YearsAtCompany": [years_at_company],
            "Education": [education],
            "MaritalStatus": [marital_status],
            "Gender": [gender]
            })
        
        input_df=pd.get_dummies(input_df,drop_first=True)
        input_df=input_df.reindex(columns=x.columns,fill_value=0)
        prob = classifier.predict_proba(input_df)[0][1]

        st.write(f"Attrition Risk Probability: **{prob:.2f}**")

        
        if prob >=best_threshold:
            st.error("Prediction: Employee is likely to LEAVE the Company.")
        else:
            st.success("Prediction: Employee is likely to STAY with the Company.")
        
 





