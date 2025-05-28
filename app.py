import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Page setup
st.set_page_config(page_title="Instagram Post Sentiment Analyser", layout="wide")

# Load BERT model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Get Instagram media post details
def fetch_ig_post_details(token, media_id):
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    params = {
        'access_token': token,
        'fields': 'caption,like_count,comments_count,timestamp,permalink'
    }
    response = requests.get(url, params=params)
    return response.json()

# Get Instagram post comments
def fetch_ig_comments(token, media_id):
    url = f"https://graph.facebook.com/v18.0/{media_id}/comments"
    params = {
        'access_token': token,
        'limit': 100
    }
    response = requests.get(url, params=params)
    return [c["text"] for c in response.json().get("data", []) if "text" in c]

# Classify sentiment
def classify_sentiment(text):
    inputs = tokenizer.encode_plus(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1)
    score = torch.argmax(probs).item()
    if score <= 1:
        return "Negative"
    elif score == 2:
        return "Neutral"
    else:
        return "Positive"

# Streamlit interface
st.title("📸 Instagram Post Sentiment Analyser")

token = st.text_input("🔐 Instagram Access Token", type="password")
media_id = st.text_input("📝 Instagram Media Post ID")

if token and media_id:
    st.success("✅ Token and Media ID entered")
    try:
        st.info("Fetching Instagram post...")
        post = fetch_ig_post_details(token, media_id)

        if "error" in post:
            st.error(f"Instagram API Error: {post['error']['message']}")
        else:
            st.subheader("🧾 Post Information")
            st.write(f"📅 Timestamp: {post.get('timestamp', 'N/A')}")
            st.write(f"📝 Caption: {post.get('caption', 'No caption')}")
            st.write(f"❤️ Likes: {post.get('like_count', 'N/A')}")
            st.write(f"💬 Total Comments: {post.get('comments_count', 'N/A')}")
            if post.get("permalink"):
                st.markdown(f"🔗 [View on Instagram]({post['permalink']})")

            # Fetch and analyze comments
            st.info("Analyzing comments...")
            comments = fetch_ig_comments(token, media_id)
            if not comments:
                st.warning("No comments found.")
            else:
                sentiments = [classify_sentiment(c) for c in comments]
                df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

                st.subheader("💬 Comment Sentiment Analysis")
                col1, col2 = st.columns([1, 2])  # Chart and table layout

                with col1:
                    st.markdown("**Sentiment Chart**")
                    st.bar_chart(df["Sentiment"].value_counts())

                with col2:
                    st.markdown("**Classified Comments**")
                    st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
else:
    st.info("Enter your access token and media post ID to begin.")
