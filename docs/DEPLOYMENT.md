# Deployment Guide

## Streamlit Community Cloud (Current)
1. Push code to GitHub.
2. Go to share.streamlit.io
3. Connect your GitHub repo.
4. Set Main file path to app.py.
5. Deploy!

## Docker (For Production)
docker build -t verdant-ai .
docker run -p 8501:8501 verdant-ai
