#%%
# from huggingface_hub import notebook_login
# notebook_login()
#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
#%%
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax
import torch
from data import load_data
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)
#%%

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = BertForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

#%%
def sentiment_analysis(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = softmax(outputs.logits, dim=1)

    sentiment_map = {0: "Negative", 1: "Positive"}
    sentiment_score, sentiment_index = torch.max(predictions, dim=1)
    sentiment_label = sentiment_map[int(sentiment_index)]

    return sentiment_label, sentiment_score.item()

# Example Usage
# text = "I love it!"
# sentiment, score = sentiment_analysis(text)
# print(f"Sentiment: {sentiment}, Confidence: {score}")

# %%
from tqdm.auto import tqdm
df = load_data()
scores = []
for i in tqdm(range(len(df))):
    text = df.iloc[i]['content']
    label, score = sentiment_analysis(text)
    scores.append(score)
df['sentiment'] = scores
df
# %%
