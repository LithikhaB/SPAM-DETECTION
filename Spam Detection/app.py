import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def load_model():
    try:
        cv = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return cv, model
    except Exception as e:
        st.error("Error loading model: " + str(e))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y= []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return y

def predict(cv, model, text):
    transformed_text = transform_text(text)
    vector_input = cv.transform([' '.join(transformed_text)])
    res = model.predict(vector_input)[0]
    return res

def main():
    cv, model = load_model()
    
    st.title("Email / SMS Spam Classifier")

    sms = st.text_input("Enter the message: ")
    
    if st.button("Predict"):
        if sms:
            res = predict(cv, model, sms)
            if res == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        else:
            st.error("Please enter a message")

if __name__ == "__main__":
    main()