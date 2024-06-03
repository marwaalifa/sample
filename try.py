from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
from streamlit_navigation_bar import st_navbar
from itertools import cycle

# read datasets
df = pd.read_csv("stud.csv")

label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

def encode_grades(g3):
    if g3 >= 18:
        return 'A'  # Excellent
    elif g3 >= 14:
        return 'B'  # Good
    elif g3 > 10:
        return 'C'  # Sufficient
    elif g3 == 10:
        return 'D'  # Pass
    else:
        return 'E'  # Fail

# Apply the encoding function to create the new 'grades' column
df['grades'] = df['G3'].apply(encode_grades)

# Verify the DataFrame and new column
print("First few rows of the DataFrame with the new 'grades' column:")
print(df.head())

# Encode the 'grades' column to numerical values for machine learning
df['grades_encoded'] = df['grades'].map({'A': 4, 'B': 3, 'C': 2, 'D': 1, 'E': 0})

# Memisahkan fitur dan label
X = df.drop(["grades", "G3"], axis=1)
y = df["grades"]

# Membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model KNN
knn_model = KNeighborsClassifier(n_neighbors=4, metric='manhattan')

# Melatih model
knn_model.fit(X_train, y_train)

# Melakukan prediksi
y_pred = knn_model.predict(X_test)

# Menghitung akurasi dan metrik lainnya
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')

# Binarize the output
y_test_bin = label_binarize(y_test, classes=[*range(len(set(y)))])
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], knn_model.predict_proba(X_test)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), knn_model.predict_proba(X_test).ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Print accuracy and precision
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")


# streamlit_navigation_bar
page = st_navbar(["Home", "Data", "Feature Importance", "KNN", "Feature Dependence", "Summary"])

###################
import streamlit as st
import pandas as pd
import numpy as np

# Assuming knn_model, df, and X are already defined and loaded elsewhere in your script

# Example metadata dictionary for features
feature_metadata = {
    "school": "Student's school. GP: Gabriel Pereira, MS: Mousinho da Silveira",
    "sex": "Student's sex. M: Male, F: Female",
    "age": "Student's age. Enter an integer value between 15 and 22.",
    "address": "Student's home address type. R: Rural, U: Urban",
    "famsize": "Family size. LE3: Less than or equal to 3, GT3: Greater than 3",
    "Pstatus": "Parent's cohabitation status. A: Apart, T: Together",
    "Medu": "Mother's education. 0: none to 4: higher education",
    "Fedu": "Father's education. 0: none to 4: higher education",
    "Mjob": "Mother's job. at_home, health, other, services, teacher",
    "Fjob": "Father's job. at_home, health, other, services, teacher",
    "reason": "Reason to choose this school. course, other, home, reputation",
    "guardian": "Student's guardian. mother, father, other",
    "traveltime": "Home to school travel time. 1: <15 min, 2: 15-30 min, 3: 30-60 min, 4: >1 hour",
    "studytime": "Weekly study time. 1: <2 hours, 2: 2-5 hours, 3: 5-10 hours, 4: >10 hours",
    "failures": "Number of past class failures. Enter an integer value between 0 and 4.",
    "schoolsup": "Extra educational support. no, yes",
    "famsup": "Family educational support. no, yes",
    "paid": "Extra paid classes. no, yes",
    "activities": "Extra-curricular activities. no, yes",
    "nursery": "Attended nursery school. no, yes",
    "higher": "Wants to take higher education. no, yes",
    "internet": "Internet access at home. no, yes",
    "romantic": "With a romantic relationship. no, yes",
    "famrel": "Quality of family relationships. 1: very bad to 5: excellent",
    "goout": "Going out with friends. 1: very low to 5: very high",
    "Dalc": "Workday alcohol consumption. 1: very low to 5: very high",
    "Walc": "Weekend alcohol consumption. 1: very low to 5: very high",
    "health": "Current health status. 1: very bad to 5: very good",
    "absences": "Number of school absences. Enter an integer value between 0 and 33.",
    "G1": "First period grade. Enter an integer value between 0 and 20.",
    "G2": "Second period grade. Enter an integer value between 0 and 20."
}

# Encoding mappings
encoding_mappings = {
    "school": {"GP": 0, "MS": 1},
    "sex": {"M": 0, "F": 1},
    "address": {"R": 0, "U": 1},
    "famsize": {"LE3": 0, "GT3": 1},
    "Pstatus": {"A": 0, "T": 1},
    "Mjob": {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "Fjob": {"at_home": 0, "health": 1, "other": 2, "services": 3, "teacher": 4},
    "reason": {"course": 0, "other": 1, "home": 2, "reputation": 3},
    "guardian": {"mother": 0, "father": 1, "other": 2},
    "schoolsup": {"no": 0, "yes": 1},
    "famsup": {"no": 0, "yes": 1},
    "paid": {"no": 0, "yes": 1},
    "activities": {"no": 0, "yes": 1},
    "nursery": {"no": 0, "yes": 1},
    "higher": {"no": 0, "yes": 1},
    "internet": {"no": 0, "yes": 1},
    "romantic": {"no": 0, "yes": 1},
}

def encode_inputs(user_inputs):
    encoded_inputs = {}
    for key, value in user_inputs.items():
        if key in encoding_mappings:
            encoded_inputs[key] = encoding_mappings[key][value]
        else:
            encoded_inputs[key] = value
    return encoded_inputs

if page == "Home":
    st.title("Student Final Grades Prediction Dashboard")
    st.markdown("""
    Welcome to the Student Grades Prediction Dashboard. This dashboard utilizes a K-Nearest Neighbors (KNN) classification model to predict student grades based on various input features. Many of these features were initially categorical (non-numeric) and have been converted to numeric values to be compatible with the model. You can find a detailed description of the feature encoding process on the `Data` Page.

    ### Why Numeric Values?
    Machine learning models require numerical input to perform calculations and make predictions. Thus, the categorical features in the [dataset](https://archive.ics.uci.edu/) are converted into numeric values. This encoding process allows the model to understand and interpret the input data effectively.

    ## Predict Students' Final Grades using KNN Classification
    ### Generate Final Score
    Input the required values for each feature and click the 'Generate Grades' button to predict the grades score.
    """)

    st.markdown("""
    #### Instructions:
    - For numerical inputs, use the provided range sliders or input boxes.
    - For categorical inputs, select the appropriate category from the dropdown menu.
    - Ensure all required fields are filled out before submitting the form.
    """)

    # Create an input form for features
    with st.form("input_form"):
        user_inputs = {}
        for column in X.columns:
             if column == 'grades_encoded':
                continue  # Skip the 'grades_encoded' column
                 
            if column in feature_metadata:
                st.markdown(f"**{column}**: {feature_metadata[column]}")
            
            # Customize input type based on each feature's data type
            if column in encoding_mappings:
                options = list(encoding_mappings[column].keys())
                user_inputs[column] = st.selectbox(f"{column}", options=options)
            else:
                user_inputs[column] = st.number_input(
                    f"{column}", 
                    min_value=int(df[column].min()), 
                    max_value=int(df[column].max()), 
                    value=int(df[column].mean()), 
                    step=1
                )
            
        submitted = st.form_submit_button("Generate Grades")
        if submitted:
            with st.spinner('Predicting...'):
                # Encode user inputs
                encoded_inputs = encode_inputs(user_inputs)
                
                # Convert encoded inputs to DataFrame
                input_df = pd.DataFrame([encoded_inputs])
                
                # Ensure column order matches model expectations
                input_df = input_df.reindex(columns=X.columns.difference(['grades_encoded']), fill_value=0)
                
                # Make prediction based on user input
                prediction = knn_model.predict(input_df)
                st.success(f"Predicted Grade: {prediction[0]}")
