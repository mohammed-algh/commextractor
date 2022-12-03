from urllib.parse import urlparse
from urllib.parse import parse_qs
from YoutubeAPI import *
import requests

# method for unshort urls
def unshorten_url(url):
    session = requests.Session()  # so connections are recycled
    resp = session.head(url, allow_redirects=True)
    return resp.url


url = str(input("Enter Youtube link: "))

urlC = unshorten_url(url)

parsed_url = urlparse(urlC)

#get the video id from the url
videoId = parse_qs(parsed_url.query)['v'][0]

startGet(str(videoId))