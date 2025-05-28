import streamlit as st
import requests
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# Page setup
st.set_page_config(page_title="Instagram Post Sentiment Analyzer", layout="wide")

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    model = BertForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Fetch Instagram media post details
def fetch_post_details(token, media_id):
    url = f"https://graph.facebook.com/v18.0/{media_id}"
    params = {
        'access_token': token,
        'fields': 'caption,like_count,permalink,timestamp'
    }
    response = requests.get(url, params=params)
    return response.json()

# Fetch all comments with pagination
def fetch_all_comments(token, media_id):
    comments = []
    url = f"https://graph.facebook.com/v18.0/{media_id}/comments"
    params = {'access_token': token, 'limit': 100}

    while url:
        res = requests.get(url, params=params if '?' not in url else {}).json()
        batch = [c["text"] for c in res.get("data", []) if "text" in c]
        comments.extend(batch)
        url = res.get("paging", {}).get("next")

    return comments

# Classify sentiment using BERT
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

# Streamlit UI
st.title("📸 Instagram Post Sentiment Analyzer")

token = st.text_input("🔐 Instagram Access Token", type="password")
media_id = st.text_input("📝 Instagram Media ID (e.g., 17970920261234567)")

if token and media_id:
    st.success("✅ Token and Media ID entered")
    try:
        st.info("Fetching post details...")
        post = fetch_post_details(token, media_id)

        if "error" in post:
            st.error(f"Instagram API Error: {post['error']['message']}")
        else:
            st.subheader("🧾 Post Information")
            st.write(f"🗓️ Created: {post.get('timestamp', 'N/A')}")
            st.write(f"📄 Caption: {post.get('caption', 'No caption')}")
            st.write(f"❤️ Likes: {post.get('like_count', 'N/A')}")
            st.markdown(f"🔗 [View Post on Instagram]({post.get('permalink', '#')})")

            st.info("Fetching and analyzing all comments...")
            comments = fetch_all_comments(token, media_id)
            if not comments:
                st.warning("No comments found on this post.")
            else:
                sentiments = [classify_sentiment(comment) for comment in comments]
                df = pd.DataFrame({"Comment": comments, "Sentiment": sentiments})

                sentiment_counts = df["Sentiment"].value_counts()
                total = len(df)
                positive_pct = (sentiment_counts.get("Positive", 0) / total) * 100
                negative_pct = (sentiment_counts.get("Negative", 0) / total) * 100
                neutral_pct = (sentiment_counts.get("Neutral", 0) / total) * 100

                st.subheader("💬 Comment Sentiment Analysis")
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("**Sentiment Distribution**")
                    st.bar_chart(sentiment_counts)

                    st.markdown(f"""
                    #### 📊 Sentiment Percentages:
                    - ✅ Positive: **{positive_pct:.1f}%**
                    - ⚠️ Neutral: **{neutral_pct:.1f}%**
                    - ❌ Negative: **{negative_pct:.1f}%**
                    """)

                with col2:
                    st.markdown("**Classified Comments**")
                    st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
else:
    st.info("Enter your access token and media ID to begin.")
