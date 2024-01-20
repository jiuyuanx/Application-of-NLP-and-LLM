#%%
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
model_name="dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

#%%
def bert_ner(text):
    # Encode text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Predict entities
    with torch.no_grad():
        outputs = model(**inputs)

    # Decode and align labels with tokens
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].tolist()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, predictions[0].tolist()):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(model.config.id2label[label_idx])
            new_tokens.append(token)

    return list(zip(new_tokens, new_labels))

# Example usage
text = "Google was founded by Larry Page and Sergey Brin while they were students at Stanford University."
entities = bert_ner(text)
print(entities)
#%%
from data import load_data
import pandas as pd

df = load_data()
df
# %%
bert_ner("""Ad sales boost Time Warner profit

Quarterly profits at US media giant TimeWarner jumped 76% to $1.13bn (£600m) for the three months to December, from $639m year-earlier.
""")
# %%
