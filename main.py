from urllib.parse import urlparse
from urllib.parse import parse_qs
from YoutubeAPI import *

url = str(input("Enter Youtube link: "))


parsed_url = urlparse(url)

#get the video id from the url
videoId = parse_qs(parsed_url.query)['v'][0]

startGet(str(videoId))