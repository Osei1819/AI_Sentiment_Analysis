import os
import pandas as pd
import numpy as np
import re
import ast
import streamlit as st
from transformers import pipeline
from dotenv import load_dotenv
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from wordcloud import WordCloud
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from langchain_community.document_loaders import TextLoader


# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# List of stop words from NLTK
stop_words = set(stopwords.words('english'))


# Load environment variables
load_dotenv()

def get_ner(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the NER if there any in the text in form of dictionary without any explation \
  and extra text.\
  #Entities to target: PPERSON, LOCATION, ORGANIZATION, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL \
  #output format: {{"word":"entity_label"}}
  here is the text: {text}
  """
  return prompt

def get_emotion_polarity(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the emtion dictionary with emotion and polarity score without any explation \
  and extra text.\
  #output format: {{
    "Happiness": "A state of well-being and contentment.",
    "Sadness": "A feeling of sorrow or unhappiness.",
    "Anger": "A strong feeling of displeasure or hostility.",
    "Fear": "An emotional response to a perceived threat or danger.",
    "Surprise": "A sudden feeling of astonishment or wonder.",
    "Disgust": "A strong feeling of dislike or disapproval.",
    "Joy": "A feeling of great pleasure and happiness.",
    "Guilt": "A feeling of responsibility or remorse for a perceived wrongdoing.",
    "Shame": "A painful feeling regarding one's own actions or behavior.",
    "Confusion": "A state of being perplexed or unclear in one's mind.",
    "Gratitude": "A feeling of thankfulness and appreciation.",
    "Regret": "A feeling of sorrow or disappointment over something that has happened.",
    "Relief": "A feeling of reassurance and relaxation after a distressing situation.",
    "Hope": "The expectation of a positive outcome or future event.",
    "Embarrassment": "A feeling of self-consciousness or shame due to a mistake or awkward situation.",
    "Contempt": "A feeling of disdain or lack of respect for someone or something.",
    "Love": "An intense feeling of deep affection.",
    "Hate": "A strong feeling of intense dislike or aversion.",
    "Frustration": "A feeling of being upset or annoyed due to being unable to achieve something.",
    "Excitement": "A feeling of great enthusiasm and eagerness." }} \
  here is the text: {text}
  """
  return prompt


def get_sentiment_polarity(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the sentiment dictionary with sentiment and polarity score without any explation \
  and extra text.\
  #output format: {{"positive":polarity score, "negative":polarity score,"neutral":polarity score }} \n
  here is the text: {text}
  """
  return prompt


def get_text_metrics(text):
  prompt = f"""
  You will receive the text which contains caribean english words. analyze it carefully. Return me the following information in dictionary without any explation \
  and extra text.\

  1. **Number of Tokens:** Calculate the total number of tokens in the text.  
  2. **Readability Score:** Assess the readability of the text using common metrics like the Flesch Reading Ease score and/or any other suitable readability metric. Provide the score only.  
  3. **Quality Score:** Evaluate the overall quality of the text on a scale of 1 to 10, considering factors like grammar, vocabulary richness, and clarity. Provide reasons for your rating.  
  4. **Tone:** Identify the tone of the text (e.g., formal, conversational, persuasive, neutral) and explain your reasoning.  
  5. **Coherence:** Assess how coherent the text is in presenting its ideas and maintaining logical flow. Provide a coherence score on a scale of 1 to 10, with an explanation.  
  6. **Intentions:** Analyze the underlying intentions or purposes of the text (e.g., to inform, persuade, entertain, or a combination of these).  


  #output format: {{
    "Number of Tokens": "score",
    "Readability Score": "score",
    "Quality Score": "score",
    "Tone": "value",
    "Coherence": "value",
    "Intentions": "value",
    }} \
  here is the text: {text}
  """
  return prompt

groq_api_key = os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

llm = ChatGroq(
  temperature=0,
  model="llama-3.1-70b-versatile",
  api_key=groq_api_key)

st.subheader("Caribbean Text Sentiment Analysis System")

file=st.file_uploader("Upload File",["csv","xlsx","pdf","txt"])

if file!=None:
  # st.write(file.name)
  file_name=file.name
  file_extension=file_name.split(".")[-1]

# # if st.button("upload"):
  if file_extension=="xlsx":
    column_name=st.text_input("Write column name which contains the Text")
    if column_name!="":
      df=pd.read_excel(file)
      df_text="".join(df[column_name.strip()].values)
      with open("df_text.txt","w") as f:
          f.write(df_text)
      # with open("df.text.txt") as f:
        
      # st.write(df_text)
      # with open(file.name, mode='wb') as w:
      #   w.write(file.getvalue())
      loader=TextLoader("df_text.txt")
      data = loader.load()
      # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
      text_chunks = text_splitter.split_documents(data)
      # st.write(text_chunks)
        
  if file_extension=="csv":
    column_name=st.text_input("Write column name which contains the Text")
    if column_name!="":
      df=pd.read_csv(file)
      df_text="".join(df[column_name.strip()].values)
      with open("df_text.txt","w") as f:
          f.write(df_text)
      # with open("df.text.txt") as f:
        
      # st.write(df_text)
      # with open(file.name, mode='wb') as w:
      #   w.write(file.getvalue())
      loader=TextLoader("df_text.txt")
      data = loader.load()
      # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
      text_chunks = text_splitter.split_documents(data)
      # st.write(text_chunks)
  
  if file_extension=="txt":
    with open(file.name, mode='wb') as w:
        w.write(file.getvalue())
    loader = TextLoader(file.name)
    data=loader.load()
    # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    text_chunks = text_splitter.split_documents(data)
  
  if file_extension=="pdf":
      with open(file.name, mode='wb') as w:
        w.write(file.getvalue())
      loader=PyPDFLoader(file.name)
      data = loader.load()
      # # split the extracted data into text chunks using the text_splitter, which splits the text based on the specified number of characters and overlap
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
      text_chunks = text_splitter.split_documents(data)
      # st.write(text_chunks[0])
    

if st.button("Analyze") and file!=None:

  sentiment_ls=[]
  polarity_ls=[]

  for i in range(len(text_chunks)):
    # Use a pipeline as a high-level helper
    # st.write(text_chunks[i].page_content)
    
    pipe = pipeline("text-classification", model="mrarish320/caribbean_english_sentiment_fine_tuned_bert")
    label=pipe(text_chunks[i].page_content)[0]["label"]
    if label=="LABEL_1":
      sentiment_ls.append("positve")
    if label=="LABEL_2":
      sentiment_ls.append("negative")
    if label=="LABEL_0":
      sentiment_ls.append("neutral")
    polarity_ls.append(pipe(text_chunks[i].page_content)[0]["score"])

  col1,col2=st.columns(2)

  with col1:
      # Create a Seaborn bar plot
      sns.set(style="whitegrid")
      fig, ax = plt.subplots(figsize=(7, 5))
      sns.countplot(x=sentiment_ls,ax=ax)
      plt.title("Sentiment Analysis")
      # plt.xticks(rotation=90)
      ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
      # plt.show()
      ax.set_title("Sentiment Analysis")
      # Display in Streamlit
      st.pyplot(fig)
 
  with col2:
      # Create a Seaborn bar plot
      sns.set(style="whitegrid")
      fig, ax = plt.subplots(figsize=(7, 5))
      sns.histplot(polarity_ls,bins=3,ax=ax)

      # sns.displot(x=emotion.keys(), y=emotion.values(),hue=emotion.keys(),ax=ax)
      plt.title("Emotion Analysis")
      # plt.xticks(rotation=90)
      ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
      # plt.show()
      ax.set_title("BERT Confidence in Sentiment Prediction")
      # Display in Streamlit
      st.pyplot(fig)

  col1,col2=st.columns(2)
  with col1:
    text = [text_chunks[i].page_content for i in range(len(text_chunks))]
    text=" ".join(text)
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=665, background_color='white',stopwords=stop_words).generate(text)
    
    # Display word cloud using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")  # Hide axes
    ax.set_title("Word Cloud Visualization", fontsize=16, color="blue")
    st.pyplot(fig)
    
  with col2:
    emotions_ls=[]
    for i in range(len(text_chunks)):
      emotion=llm.invoke(get_emotion_polarity(text_chunks[i].page_content)).content
      emotion=ast.literal_eval(emotion)
      max_key = max(emotion, key=emotion.get)  # Get the key with the maximum value
      # max_value = data[max_key]  # Get the corresponding value
      emotions_ls.append(max_key)
    # st.write(emotions_ls)

    # Create a Seaborn bar plot
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=emotions_ls,ax=ax)
    plt.title("Emotion Analysis")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # Display in Streamlit
    st.pyplot(fig)



  col1,col2=st.columns(2)
  with col1:
    ner_ls=[]
    for i in range(len(text_chunks)):
      ner=llm.invoke(get_ner(text_chunks[i].page_content)).content
      ner_ls.extend(ast.literal_eval(ner).values())
    # Convert the ner dictionary to a Pandas DataFrame for long-form data
    ner_df = pd.DataFrame({'entity_label': ner_ls})
    fig, ax = plt.subplots(figsize=(7, 5))
    # # Now, use the 'entity_label' column for both x and hue
    sns.countplot(x='entity_label', hue='entity_label', data=ner_df,ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # # plt.show()
    ax.set_title("NER Analysis")
    # # Display in Streamlit
    st.pyplot(fig)


  with col2:
    text_metrics_ls=[]
    metrics_names=[]
    metrics_values=[]
    readability_score=0
    quality_score=0
    coherence=0
    for i in range(len(text_chunks)):    
      text_metrics=llm.invoke(get_text_metrics(text_chunks[i].page_content)).content
      text_metrics=ast.literal_eval(text_metrics)
      readability_score+=text_metrics["Readability Score"]
      quality_score+=text_metrics["Quality Score"]
      coherence+=text_metrics["Coherence"]
      
      metrics_names.append(text_metrics["Tone"].strip().lower())
            
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=["Readability Score","Quality Score","Coherence"],y=[readability_score/len(text_chunks),quality_score/len(text_chunks),coherence/len(text_chunks)],hue=["Readability Score","Quality Score","Coherence"],ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # plt.show()
    ax.set_title("Text Metrics")
    # Display in Streamlit
    st.pyplot(fig)


  # col1,col2=st.column
  fig, ax = plt.subplots(figsize=(7, 5))
  sns.countplot(x=metrics_names,ax=ax)
  ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
  ax.set_title("Tone Analysis")
  # Display in Streamlit
  st.pyplot(fig)    
  


