import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
# for importing data using pandas library 
data = pd.read_csv("D:\spam.csv")

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

mess = data['Message']
cat = data['Category']

(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#Creating Moddel
model = MultinomialNB()
model.fit(features, cat_train)

#Test Model
features_test = cv.transform(mess_test)
#print(model.score(features_test, cat_test))

# Add sample spam/ham messages
examples = {
    "Spam Example": "WINNER!! You've won a $1000 gift card! Click here >>>",
    "Not Spam Example": "Hey, are we meeting for lunch today?"
}

st.subheader("Try these examples:")
example_choice = st.selectbox("Choose an example:", list(examples.keys()))
if st.button("Load Example"):
    input_mess = examples[example_choice]
    st.text_area("Example Message", input_mess)
    

#predict the test data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

st.header("Spam Detection")

input_mess = st.text_input("Enter the message Here")

if st.button('Predict'):
    output = predict(input_mess)
    st.success(output[0])
