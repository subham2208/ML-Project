# import libraries 
import pandas as pd
import streamlit as st 
import pickle 
from pathlib import Path

load_lr = pickle.load(open('lr_classifier.sav',"rb"))

vectorize = pickle.load(open('lr_vectorizer.sav', "rb"))

def news_prediction(news):
  # new_news = "LONDON (Reuters) - LexisNexis, a provider of l."
  testing_news = {"text":[news]}
  new_def_test = pd.DataFrame(testing_news)
  new_x_test = new_def_test['text']
  new_tfidf_test = vectorize.transform(new_x_test)
  pred_dt = load_lr.predict(new_tfidf_test)
  
  if (pred_dt[0] == 0):
    return "This is Fake News!"
  else:
    return "The News seems to be True!"


    
def main():
  
  st.title("Fake News Prediction System")
  st.write("""## Input your News Article down below: """)
  user_text = st.text_input('')
  if st.button("Predict"):
    news_pred = news_prediction(user_text)
    
    if (news_pred == "This is Fake News!"):
      st.error(news_pred, icon="🚨")
    else:
      st.success(news_pred)
      st.balloons()

  
if __name__ == "__main__":
  main()