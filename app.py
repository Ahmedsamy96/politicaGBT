import streamlit as st
import os
try:
    from PIL import Image
except ImportError:
    import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import bs4
import string
import requests
import datetime
from datetime import date
import pytesseract
from deep_translator import GoogleTranslator
import streamlit.components.v1 as components
import openai as ai
import snscrape.modules.twitter as sntwitter
import fontstyle


PAGE_CONFIG = {"page_title":"PoliticAGBT","page_icon":"https://qph.cf2.quoracdn.net/main-thumb-1299752418-200-hmhzoxrsjjrxxwidtrmqvndctyvbgvud.jpeg","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)



def Topic_modeler(document_content,prompt):
    # Replace YOUR_API_KEY with your OpenAI API key
    ai.api_key = "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"

    model_engine = "text-davinci-003"
    #print(prompt)
    prompt = prompt + str( {document_content} )

    # Set the maximum number of tokens to generate in the response
    max_tokens = 1024

    # Generate a response
    completion = ai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )


    ChatGBT_output = str(completion.choices[0].text)
    return ChatGBT_output

####################################################################################################################################################

# Get the key from an environment variable on the machine it is running on

def main():
    st.title("ChatGPT Mastery")
    st.subheader("Just Upload your new Document, then you can get a high level of text analytics by making high use of ChatGBT ")

    with st.sidebar.container():
		    st.image("https://qph.cf2.quoracdn.net/main-thumb-1299752418-200-hmhzoxrsjjrxxwidtrmqvndctyvbgvud.jpeg", use_column_width=True)

    menu = ["Home","About"]
    choice = st.sidebar.selectbox('Menu',menu)
    if choice == 'Home':
        st.subheader("Main Page")



    uploaded_file = st.file_uploader("Upload a file" , type=("png", "jpg","jpeg"), accept_multiple_files=False)
    if uploaded_file:
        with st.spinner("Applying OCR on the img...."):
            # Upload Document Image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

            # Apply OCR on the image
            custom_config = r'-l eng+ara --psm 6'
            extractedInformation = pytesseract.image_to_string(image, config=custom_config)
            Arabic_text = ' '.join(re.findall(r'[\u0600-\u06FF0-9_]+',extractedInformation))
            Arabic_text = fontstyle.apply(Arabic_text, 'Sans-Serif')
            # translate to English as Spacy not defined with arabic lang
            translated_doc = GoogleTranslator(source='auto', target='en').translate(Arabic_text)
            Arabic_text = GoogleTranslator(source='auto', target='ar').translate(translated_doc)
  


        # Append the new document to the main dataframe
        st.subheader("Here is the OCR result of your uploaded image.")
        options = ['English Document', 'Arabic Document']
        selection = st.sidebar.selectbox("Choose OCR result's Language", options , index=1)
        if selection == 'English Document':
            st.success(translated_doc, icon="✅")
        elif selection == 'Arabic Document':
            st.success(Arabic_text, icon="✅")

        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)


        prompt1 = f"Give me 3 variations of a topic for this text in the fewest possible words"
        ChatGBT_output = Topic_modeler(translated_doc,prompt1)
        lines = ChatGBT_output.splitlines()        
        list_of_Topics = [i[2:] for i in lines[2:]]
        list_of_Topics_arabic = [GoogleTranslator(source='auto', target='ar').translate(ele) for ele in list_of_Topics]
        st.subheader("List of 3 Topics express Your given text. (ChatGBT Powered)")
        #st.write(list_of_Topics_arabic)
        for i in list_of_Topics_arabic:
            st.markdown("**-** " + i)

        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)


        st.subheader("A wide analytical vision describing your content based on related actions. (ChatGBT Powered)")
        prompt3 = "Expalin this text with highly analytics sence including last actions related to this topic and Finally give me your recommentations about this topic, here is the text : "
        Text_analytics = Topic_modeler(translated_doc ,prompt3)
        st.success(str(Text_analytics), icon="✅")

        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)

        st.subheader("Predicting the First and most relevant title's Class. (ChatGBT Powered)")
        prompt2 = "I'll give you a text to be classified into only one of these categories: ( 'Russian War' , 'Renaissance Dam crisis' , 'The global economic crisis' ), here is the text : "
        MainTopic_Class = Topic_modeler(list_of_Topics[0] ,prompt2)
        st.title( str(MainTopic_Class) )

        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)


        today = date.today()
        search_term = str(MainTopic_Class)
        Month_age = (today - datetime.timedelta(30)).strftime('%Y-%m-%d')

        tweets_list = []
        replyCount_list = []
        retweetCount_list = []
        likeCount_list = []
        #locations = []
        with st.spinner("Scapping last 100 Tweets through a month based on the above Topic as a test sample..."):
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f"{search_term} since:"+ Month_age +"until:"+str(today)).get_items()):
                if i == 100:
                    break
                Tweet = tweet.content
                Tweet = re.sub(r'https?:\/\/\S*', '', tweet.content, flags=re.MULTILINE) #Strips out links, comment out if you want links
                Tweet = [re.sub('@[\w]+ ','',Tweet)] #Strips out @'s, comment out if you want @'s
                tweets_list.append(Tweet[0]) 
                replyCount_list.append(tweet.replyCount)
                retweetCount_list.append(tweet.retweetCount)
                likeCount_list.append(tweet.likeCount)
                #locations.append(tweet.coordinates) 

        st.subheader("Scrapped Dataset based on the predicted Topic of your text (Month Ago).")
        # Creating a dataframe from the tweets list above 
        #tweets_df = pd.DataFrame(tweets_list, columns=['tweets'])
        tweets_df= pd.DataFrame({'Reply_Count': replyCount_list,'Retweet_Count': retweetCount_list,'Like_Count': likeCount_list ,'Tweet_Content':tweets_list})
        st.write(tweets_df)


        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)


        # Sentiment analysis Column creation
        #with st.spinner("Generating Sentiment analysis Column using ChatGBT.."):
        tweets_df = tweets_df.iloc[:20]
        st.subheader("Generating Sentiment analysis Column...")
        prompt4 = "Classify the given text to be only one of these classes : ( 'Positive' - 'Negative' - 'Neutral' ) , here is the text: "
        tweets_df['GPT_SentimentAnalysis'] = tweets_df['Tweet_Content'].apply(lambda x: str(Topic_modeler(x ,prompt4)).strip() )
        st.write(tweets_df)
        
        @st.cache_data
        def convert_df(df):
          return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(tweets_df)
        st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv' )

        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)


        st.subheader("Sample distribution of the sentimental status in the last month on **Twitter** about your Topic.")
        st.bar_chart(tweets_df.GPT_SentimentAnalysis.value_counts())

        components.html("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """)

if __name__ == '__main__':
	main()
