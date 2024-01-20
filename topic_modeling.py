#%%
from data import load_data
import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# %%
def preprocess_text(doc):
    # Tokenize, remove stopwords, and lowercase
    stop_words = set(stopwords.words('english'))
    return [word for word in gensim.utils.simple_preprocess(doc) if word not in stop_words]

#%%
df = load_data()
df

# %%
df['processed_docs'] = df['content'].map(preprocess_text)

# Create a dictionary representation of the documents
dictionary = corpora.Dictionary(df['processed_docs'])

# Filter out words that occur in less than 20 documents, or more than 50% of the documents
dictionary.filter_extremes(no_below=20, no_above=0.5)

# Create a bag-of-words representation of the documents
corpus = [dictionary.doc2bow(doc) for doc in df['processed_docs']]

# Set up the LDA model
# Assuming we want to find 5 topics
lda_model = gensim.models.LdaMulticore(corpus, num_topics=5, id2word=dictionary, passes=10, workers=2)

# Print the topics
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
# %%
def format_topics_sentences(ldamodel, corpus):
    # Init output
    topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                topics_df = topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    return topics_df

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus)

# Concatenate the original DataFrame with the topics DataFrame
df = pd.concat([df.reset_index(drop=True), df_topic_sents_keywords.reset_index(drop=True)], axis=1)
# %%
df
# %%
