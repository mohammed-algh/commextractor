import pyarabic
import re
import demoji

# Removing punctuations like . , ! $( ) * % @
# Removing URLs
# Removing Stop words
# Lower casing
# Tokenization
# Stemming
# Lemmatization

# https://github.com/saobou/arabic-text-preprocessing



def remove_emojis(data):
    return demoji.replace(data, "")

def cleaner(text):
    text = str(text).replace('[', ' ').replace(']', ' ')
    text = str(text).replace("'", ' ').replace("'", ' ')
    text = re.sub('<a[\s\S]*<[\s\S]a>',' ',text)
    text = re.sub('<br>', ' ', text)
    return text.strip()

