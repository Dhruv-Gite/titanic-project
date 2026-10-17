import joblib
import streamlit

model = joblib.load('lregration.pkl')
knn = joblib.load('knn.pkl')
svm = joblib.load('svm.pkl')

age = int(input('Enter Your Age'))

output = (model.predict([[age]]))

if output == 1:
    print('Survived')
else:
    print('Did Not Survived')