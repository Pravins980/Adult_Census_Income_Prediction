import pickle
import numpy as np
import pandas as pd
import streamlit as st


# Load Model
model = pickle.load(open('model/model.pkl', 'rb'))
dataset = pd.read_csv('data/processed/adult.csv')

from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Census Income Prediction", page_icon="🤖")

st.title("Census Income Prediction")
st.markdown(
    "Census Income Prediction Dashboard")

le = LabelEncoder()

dataset['income'] = le.fit_transform(dataset['income'])

dataset = dataset.replace('?', np.nan)

columns_with_nan = ['workclass', 'occupation', 'native.country']

for col in columns_with_nan:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)

for col in dataset.columns:
    if dataset[col].dtypes == 'object':
        encoder = LabelEncoder()
        dataset[col] = encoder.fit_transform(dataset[col])

X = dataset.drop('income', axis=1)
Y = dataset['income']

X = X.drop(['workclass', 'education', 'race', 'sex',
            'capital.loss', 'native.country', 'fnlwgt', 'relationship',
            'capital.gain'], axis=1)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

def predict(marital_name,age_value,edu_num_value,occupation_value,hours_value):
    # if request.method == 'POST':
    marital_value = 0

    if marital_name == 'Married-civ-spouse':
        marital_value = 1
    elif marital_name == 'Never-married':
        marital_value = 2
    elif marital_name == 'Divorced':
        marital_value = 3
    elif marital_name == 'Separated':
        marital_value = 4
    elif marital_name == 'Widowed':
        marital_value = 5
    elif marital_name == 'Married-spouse-absent':
        marital_value = 6
    elif marital_name == 'Married-AF-spouse':
        marital_value = 7

    features = [age_value, edu_num_value, marital_value,
                occupation_value, hours_value]

    int_features = [int(x) for x in features]
    final_features = [np.array(int_features)]
    prediction = model.predict(scaler.transform(final_features))

    if prediction == 1:
        st.write("Income is more than 50K")
    elif prediction == 0:
        st.write("Income is more than 50K")

    # return render_template('index.html', prediction_text='{}'.format(output))



with st.form("my_form"):
    Marital = st.selectbox('Marital Status', (
    'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent',
    'Married-AF-spouse'))
    Age = st.text_input(
        label="Age",
        placeholder="Age",
    )
    Education = st.text_input(
        label="Years of Education",
        placeholder="Years of Education",
    )
    Occupation = st.text_input(
        label="Occupation Code",
        placeholder="Occupation Code",
    )
    Hours = st.text_input(
        label="Hours of work per week",
        placeholder="Hours of work per week",
    )


    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write('Inputs:')
        st.write('Marital Status:',Marital, 'Age:',Age, 'Education:',Education,'Occupation:', Occupation,'Hours:', Hours)
        predict(Marital, Age, Education, Occupation, Hours)


if __name__ == "__main__":
    print('test')