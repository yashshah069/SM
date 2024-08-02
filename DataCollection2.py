import pandas as pd 
import requests
from textblob import TextBlob

# Set YouTube video ID, maximum number of comments to retrieve, and API key 
video_id = "Q33TkQKlIMg"
max_result = 50
api_key = "AIzaSyC_4xZTiNuz1O-Qu5kYnlg82riP30KRIxY"

# Retrieve video information 
video_info_url = f"https://www.googleapis.com/youtube/v3/videos?part=id%2Csnippet&id={video_id}&key={api_key}"
video_info_response = requests.get(video_info_url) 
video_info_data = video_info_response.json()

# Retrieve video comments 
comments_url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&videoId={video_id}&part=snippet&maxResults={max_result}"
comments_response = requests.get(comments_url) 
comments_data = comments_response.json()

# Extract comments from the comments_data JSON 
comments = []
for item in comments_data['items']:
    comment = item['snippet']['topLevelComment']['snippet']['textOriginal'] 
    comments.append(comment)


# Define function to perform sentiment analysis on a given comment 
def get_comment_sentiment(comment):
    analysis = TextBlob(comment)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity == 0:
        return "Neutral"
    else:
        return "Negative"
# Perform sentiment analysis on all comments 
comment_list = []
sentiment_list = []
for comment in comments:
    sentiment = get_comment_sentiment(comment) 
    comment_list.append(comment) 
    sentiment_list.append(sentiment) 
    print(f"{comment} : {sentiment}")
# Create DataFrame from comments and sentiments
sentiment_df = pd.DataFrame({"Comments": comment_list, "Sentiment": sentiment_list}) 
# Save DataFrame to a CSV file 
sentiment_df.to_csv("YouTube_Comments_Sentiment.csv", index=False)
