# import streamlit as sm
# import streamlit.components.v1 as com
# from Preprocessing import doPreprocessing
# from YoutubeAPI import startGet,all_comments
# from main import unshorten_url
#
# sm.title("WASI | Arabic Sentiment Analyzer")
#
# text = sm.text_input("Enter a Text")
#
# startGet(text)
#
# for i in all_comments:
#
#     sm.write("""Text After Pre-processing: """,preprocessing(i))