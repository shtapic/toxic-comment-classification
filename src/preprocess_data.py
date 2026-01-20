import pandas as pd
import re

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['comment_text'] = df['comment_text'].apply(clean_text)
    return df



def clean_text(text: str) -> str:
    """Clean the input text by lowering case and removing special characters."""

    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " <URL> ", text)
    text = re.sub(r"\d+", " <NUMB> ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

