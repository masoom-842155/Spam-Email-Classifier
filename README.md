# Spam-Email-Classifier

Building a Simple Spam Detection Model with Python and Streamlit
Spam messages are an ongoing challenge for users of email and messaging platforms. With the rise in spam, effective detection is key to filtering unwanted content. In this blog post, we'll explore how to build a simple spam detection model using Python, machine learning, and Streamlit, a framework for building web applications. We’ll break down the code, explain the steps involved, and show you how you can easily deploy this model using Streamlit.

Let’s get started!

Prerequisites:

Before diving into the code, make sure you have the following Python libraries installed:
pandas: For data manipulation.
scikit-learn: For machine learning tasks like model training and evaluation.
streamlit: For building interactive web apps.

You can install them using pip:
pip install pandas scikit-learn streamlit

The Code Breakdown
Let’s walk through the code to understand how it works.

Step 1: Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

Here, we import the necessary libraries:

pandas is used for loading and manipulating the dataset.
train_test_split helps to split the dataset into training and testing sets.
CountVectorizer is used to convert the text data into numerical features that our model can understand.
MultinomialNB is the Naive Bayes classifier we’ll use to classify the messages as either spam or not spam.
streamlit is used for creating a simple web app where users can input messages and receive predictions.

Step 2: Load and Preprocess Data

data = pd.read_csv("D:/spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
The dataset is loaded using pandas.read_csv(). The CSV file (spam.csv) should contain two columns: Message (the text) and Category (whether it's spam or not).
We remove any duplicate rows using drop_duplicates().
The Category column, which originally contains "ham" and "spam", is replaced with "Not Spam" and "Spam" for better clarity.

Step 3: Split the Data

mess = data['Message']
cat = data['Category']
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

We split the dataset into features (mess) and labels (cat).
Then, we use train_test_split to create training and testing sets. 80% of the data is used for training, and 20% is used for testing the model.

Step 4: Convert Text Data to Numerical Features

cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

We use CountVectorizer to convert the text messages into a matrix of token counts. This step transforms each word in the message into a feature that the model can understand. The stop_words='english' argument removes common English stop words (e.g., "and", "the") from the text to reduce noise.

Step 5: Build and Train the Model

model = MultinomialNB()
model.fit(features, cat_train)

We create a MultinomialNB model, which is suitable for text classification tasks like spam detection.
The model is trained using the fit() function with the training data.

Step 6: Test the Model

features_test = cv.transform(mess_test)
#print(model.score(features_test, cat_test))

The test data is transformed using the same CountVectorizer instance to ensure the same feature extraction process as the training data.
While we commented out the model.score() line, it would typically be used to evaluate the accuracy of the model on the test data.

Step 7: Create a Web App with Streamlit

Streamlit allows us to quickly turn our machine learning model into an interactive web application. The following code creates an interface where users can input messages, and the app predicts whether the message is spam or not.

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

Here, we provide two example messages: one spam and one not spam. Users can select an example from the dropdown, and it will be displayed in a text area.

Step 8: Prediction Function

def predict(message):
    input_message = cv.transform([message]).toarray()
    result = model.predict(input_message)
    return result

This function takes a message as input, transforms it into numerical features using the same CountVectorizer, and uses the trained model to predict whether the message is spam or not.

Step 9: Final Streamlit Interface

st.header("Spam Detection")

input_mess = st.text_input("Enter the message Here")

if st.button('Predict'):
    output = predict(input_mess)
    st.success(output[0])

A text input field allows the user to type a message.
When the "Predict" button is pressed, the message is passed to the predict() function, and the prediction is displayed.

Running the App

To run this app, save the code in a Python file, e.g., spam_detection.py, and then execute the following command in the terminal:

streamlit run spam_detection.py

The Streamlit app will open in your browser, and you can start interacting with it, entering your own messages to predict if they are spam or not.

Conclusion
In this blog post, we built a simple spam detection model using Naive Bayes and integrated it with Streamlit to create an interactive web application. This combination of machine learning and web app deployment allows you to quickly turn your ideas into actionable tools. Whether you're interested in building more complex models or just want to create interactive tools for data analysis, Streamlit makes it easy and fun to get started.

























