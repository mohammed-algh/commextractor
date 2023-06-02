from googleapiclient.discovery import build
from Preprocessing import *
import csv
import re

api_key = 'AIzaSyCeOTkJfH0_XNhpzeVg3zrDF3Xetgjbt9w'

#{3}


# recursive function to get all comments
def get_comments(youtube, video_id, next_view_token):
    global all_comments

    # check for token
    if len(next_view_token.strip()) == 0:
        all_comments = []
    try:
        if next_view_token == '':
            # get the initial response
            comment_list = youtube.commentThreads().list(part = 'snippet', maxResults = 100, videoId = video_id, order = 'relevance').execute()
        else:
            # get the next page response
            comment_list = youtube.commentThreads().list(part = 'snippet', maxResults = 100, videoId = video_id, order='relevance', pageToken=next_view_token).execute()
    except Exception as error:
        if "commentsDisabled" in str(error):
            print("Comment of this video disabled")
            print()
        if "videoNotFound" in str(error):
            print("Video not found")
            print()

    # loop through all top level comments
    authList = []
    for comment in comment_list['items']:

        # add comment to list
        if len(removeSymbolNoise(str([comment['snippet']['topLevelComment']['snippet']['textDisplay']]))) >=7 :
            try:
                authID = [comment['snippet']['topLevelComment']['snippet']['authorChannelId']['value']]
                authComm = [comment['snippet']['topLevelComment']['snippet']['textDisplay']]
            except:
                pass

            # to remove author repeated comments.
            if authID not in authList and not bool(re.search('[a-zA-Z]+', str(authComm))) and len(str(authComm)) >= 20:
                authList.append(authID)

                # call preprocessing function in YoutubeAPI file
                authClComm = preprocessing(authComm)
                if len(authClComm) >= 3:
                    all_comments.append(authClComm.strip())

        #{2}

    if "nextPageToken" in comment_list:
        return get_comments(youtube, video_id, comment_list['nextPageToken'])
    else:
        return []


all_comments = []

# build a youtube object using our api key
def startGet(video_id):
    yt_object = build('youtube', 'v3', developerKey=api_key)

    # get all comments
    comments = get_comments(yt_object, video_id, '')

   #{1}

    count = 0

    # store the result in csv file

    # with open('datasetFinal.csv', 'a', newline='', encoding="utf-8-sig") as f:
    #     csvwriter = csv.writer(f)
    #
    #     for comment in all_comments:
    #         csvwriter.writerow([str(comment)])

    # print result in the terminal
    for comment in all_comments:

        print(str(comment))

        #count number of the comments
        count = count + 1

        #new line
        print()

    print("-----------------------")
    print("Total comments: "+str(count))
    print("-----------------------")