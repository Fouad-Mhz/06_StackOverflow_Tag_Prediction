
"""
In this notebook we extract data from stackoverflow questions and perform an exploratory data analysis of the dataset.

The data can be originally found at: https://data.stackexchange.com/stackoverflow/query/new

# Download data: 100k posts containining Title, tags and text and score higher that 20

We create an SQL program that samples 50K random posts without repetition. 
To achieve efficiency if we wish to sample 50k more posts, we shall sample independently from the previous 50k sample such that the merge of the samplings must be filtered for duplicates.
Here we use a dataset merge of 2 samplings with score greater than 20, as discussed in the data analisys notebook.
"""

# imports

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# downlaod and unzip tags counts
from google_drive_downloader import GoogleDriveDownloader as gdd

files = ['1aGnT5OqAN7KmJoyEiWnFsnCELVun7s-s','1-aKDsJDXPkhpzU0sAZ_4SC3gXKzGYuJV', '1WC60yk8Gen_eQocVPhnYNd3A5NJ5qZUf']

for file in files:
  data_path = os.path.join('.','data')

  gdd.download_file_from_google_drive(file_id=file,
                                      dest_path=data_path,
                                      unzip=True)
  # remove .zip data
  os.system('rm -rf data')

"""# 1. Data pre-processing:

First we need to analyse the data structure and decide which processing will be applied to the dataset.

"""

df_part1 = pd.read_csv('QueryResults_part1.csv')
df_part2 = pd.read_csv('QueryResults_part2.csv')

"""Since the two dataframes were gathered at random separately we must join without repeating data."""

data = pd.concat([df_part1,df_part2]).drop_duplicates().reset_index(drop=True)

"""As a sanity check, the joint dataset should have less than 100k questions."""

len(data)

"""Now we split our dataset into training and testing data. We can split training data further into another training set and an evaluation set.

To avoid any bias in this sampling we shufle the dataset...
"""

from sklearn.model_selection import train_test_split

df, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True,) # fix random state to reproduce results on other platforms

len(df)

len(test)

"""## 1.1 Tag pre-processing

Tags are written between less-than and greather-than signs on a single string.To work with individual tags we first have to process them using regex and create a list of tags that can be manipulated.
"""

# separate tags into a list of tags using a lambda function
get_tags = lambda x: re.findall("\<(.*?)\>", x)

df['Tags'] = df['Tags'].apply(get_tags)
test['Tags'] = test['Tags'].apply(get_tags)

df.head()

"""#### Get Stackoverflow full tag data of top 50k most frequent tags

As discussed in the exploratory analysis we can use stackoverflow top 50k tags of the whole dataset and obtain the mapping from token to tags as explained in the exploratory analysis.
"""

df_tags = pd.read_csv('TopTags.csv')
df_tags.at[553, 'TagName'] = 'null'
df_tags.at[1819, 'TagName'] = 'nan'

# we will use the tag array to check for tag matches
tags_array = (df_tags.TagName).to_numpy()[:1000]

"""# 2. Title Pre-processing

Here we process the title into a bow representation. This allows for a more in depth analysis of our data such that we can process it into features that can be better dealt by machine learning algorithms.
"""

# Transform titles to list
df.Title.to_list()[:10]

# download and import required packages
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import nltk.stem as stemmer
from nltk.stem.porter import *
import nltk
from nltk import word_tokenize
from nltk.tokenize import MWETokenizer

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# instantiate stemmer that will be used along the processing pipeline
stemmer = PorterStemmer()

"""**Observation:** Since we use a out of the box tokenizer we can add our own rules to it. An example would be to add the token for C# (c sharp). Otherwise it would be removed by the tokenizer."""

# add exceptions to tokenizer
tokenizer = nltk.tokenize.MWETokenizer()
tokenizer.add_mwe(('c', '#'))

"""Here we define the preprocessing we will apply to text."""

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    # remove ponctuation but keep relevant data
    initial_preprocess = lambda text : "".join([char for char in text if char not in '!"$%&\'()*,./:;<=>?@[\\]^_`{|}~']).lower()
    tokens = tokenizer.tokenize(word_tokenize(initial_preprocess(text)))
    for token in tokens:
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(lemmatize_stemming(token))
    return result

"""And we apply it to the train and test dataset."""

processed_titles = df.Title.map(preprocess)
test_processed_titles = test.Title.map(preprocess)

"""Below we observe that the result of the tokenization."""

processed_titles

"""Now we generate a corpora dictionary using the tokenized sentences from the train dataset. We will print the first 10 words from the dictionary."""

# Generate corpora dictionary
title_dictionary = gensim.corpora.Dictionary(processed_titles)

"""Print the first 10 entries"""

print('there are {} entries on the corpora dict.\nFirst 10 entries:'.format(len(title_dictionary)))
print(list(title_dictionary.values())[:10])

"""To reduce the computational complexity we filter out words that appear less than a fixed number of times. The documentation of gives the following parameters:

* no_below (int, optional) – Keep tokens which are contained in at least no_below documents.

* no_above (float, optional) – Keep tokens which are contained in no more than no_above documents (fraction of total corpus size, not an absolute number).

* keep_n (int, optional) – Keep only the first keep_n most frequent tokens.

* keep_tokens (iterable of str) – Iterable of tokens that must stay in dictionary after filtering.

We observe a great reduction in the number of the dictionary entries. This will greatly accelerate computations.
"""

title_dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n=10000)
print('After filtering there are {} entries on the corpora dict.\nFirst 10 entries:'.format(len(title_dictionary)))
print(list(title_dictionary.values())[:10])

"""With the dictionary in hands we can now create a BoW representation of titles. We do this both for the train and test sets."""

# create bow of title filtered corpus
title_bow_corpus = [title_dictionary.doc2bow(title) for title in processed_titles]
test_title_bow_corpus = [title_dictionary.doc2bow(title) for title in test_processed_titles]

"""We can now observe the result of the preprocessing on a string of text"""

# observe pre processing result on a sampling of a given dataset
def sample_nlp_pipeline(sample_idx, dataframe, bow_corpus):
  print('sample idx:', sample_idx)

  print('sample tags:', dataframe.Tags.to_list()[sample_idx])
  print('\nprocessing pipeline: \n')
  print('sample title:', dataframe.Title.to_list()[sample_idx])
  print('preprocessed title:', processed_titles[sample_idx])
  print('bow_corpus of title:', bow_corpus[sample_idx])
  print('\nbag of words equivalence: ')
  bow_doc_sample = bow_corpus[sample_idx]
  for i in range(len(bow_doc_sample)):
      print("Word {} (\"{}\") appears {} time.".format(bow_doc_sample[i][0],
                                                title_dictionary[bow_doc_sample[i][0]], bow_doc_sample[i][1]))

"""Let's observe the result on a random question"""

# test preprocessing on a random question
sample_nlp_pipeline(np.random.randint(len(df.Tags)), df, title_bow_corpus)

"""## 3.1 Token2tag

To perform classification and evaluate we will use a dict of tokens to tag, as explained in the exploratory analysis notebook.
"""

df_tags.head()

df_tags['tokenized'] = df_tags.TagName.apply(preprocess)

# get array with tags
tags_array = (df_tags.TagName).to_numpy()

# get an array with tokenized tags
tokenized_tags = df_tags.tokenized.to_numpy()

# get an array with the tag count
full_tag_count = (df_tags.Count).to_numpy()

def select_first(token_list):
  return [token_list[0]]

# count number of total tags
total = sum(full_tag_count)
# initialize the
n_lost = 0

# we will capture index of divided or lost tags
eliminated_tags = []
divided_tags = []

for idx, tag in enumerate(tags_array):
  tokenized_tag = preprocess(tag)
  # check if tag was mapped to zero
  if len(tokenized_tag) == 0:
      eliminated_tags.append(idx)
      n_lost += full_tag_count[idx] / total * 100
          #print("The tag '{}' ({:.2f}% of tags) was eliminated by tokenization".format(full_tags_array[idx], full_tag_count[idx]/total*100))


  # check if tags were divided
  if len(tokenized_tag) > 1:
      n_lost += full_tag_count[idx] / total * 100
      divided_tags.append(idx)
          #print("The tag '{}' ({:.2f}% of tags) was divided by tokenization into '{}'".format(full_tags_array[idx], full_tag_count[idx]/total*100, tokenized_tag))

print('\nTotal ammount of lost tags for tokenization: {:.2f}%'.format(n_lost))


df_tags.loc[divided_tags, 'tokenized'] = df_tags.loc[divided_tags, 'tokenized'].apply(select_first)

token2tag_dict = {}
for tag, token in zip(df_tags.TagName.to_numpy(), df_tags.tokenized.to_numpy()):
  # token should be a list containing only one element
  if len(token) == 0:
    pass
    #print('passed {} because the token is null'.format(tag))
  elif token[0] in token2tag_dict.keys():
    pass
    #print('passed {} because the token has already been mapped'.format(tag))
  else:#returning-a-view-versus-a-copy https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html
    token2tag_dict[token[0]] = tag

"""# 3. Unsupervised learning on titles

In this section we will train a LDA model with the pre-processed titles. The resulting model will cluster the questions according to a number of topics.

LDA stands for Latent Dirichlet Allocation. It is a probabilistic generative model allowing to explain sets of observations, by means of unobserved groups, them themselves defined by data similarities.

In our context, since we already verified in the exploratory analysis that tags are present in the titles and text of questions, we can make the hypthesis that they end up making part of the topics.
Furthermore, since the tags seem to appear on separate clusters of questions (ie. a python tag is not present in a C question), we also can expect that the topics encoding will also preserve this separation.

## 4.1 Training

### Question: How many topics should we use?

This is an hyperparameter of the model. Tweaking it would offer ways to improve the results if we so desired.
"""

import os
# Get the number of available CPU cores
num_cpus = os.cpu_count()
print("Number of CPU cores available:", num_cpus)

# Fit LDA model using preprocessed titles

title_lda_model = gensim.models.LdaMulticore(
    title_bow_corpus,
    num_topics=100,
    id2word=title_dictionary,
    passes=10,
    workers=2,
    eta='auto'
)

from google.colab import drive
drive.mount('/content/drive')

# Save the model
model_file_path = "/content/drive/MyDrive/Colab Notebooks/Lda models/title_lda_model"
title_lda_model.save(model_file_path)

"""The topics are a weighted combination of the processed tokens. Our unsupervised approach is now able to use this feature without any supervision to generate the predictions. We expect that the tags are present among the topic composition with high confidence."""

# Print all topics and their token composants
for idx, topic in title_lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

"""## 4.3 Inference

Now we test the hypothesis of tags appearing in the themes:
"""

# Given the index of the question print the pre processing pipeline and the score
def infer_topic_score(sample_idx, bow_corpus):
  for index, score in sorted(title_lda_model[bow_corpus[sample_idx]], key=lambda tup: -1*tup[1]):
      print("\nScore: {}\t \nTopic: {}".format(score, title_lda_model.print_topic(index, 10)))

"""Indeed we observe that the tags are present in at least some of the topics!"""

idx = 1
print("###################Processing#####################")
sample_nlp_pipeline(idx, df, title_bow_corpus)

print("###################Inference######################")
infer_topic_score(idx, title_bow_corpus)

idx = 11010
print("###################Processing#####################")
sample_nlp_pipeline(idx, df, title_bow_corpus)
print("\n###################Inference######################")
infer_topic_score(idx, title_bow_corpus)

"""We can also test on unseen data."""

# test on unseen data
unseen_title = 'How can I declare a struct in java'
bow_vector = title_dictionary.doc2bow(preprocess(unseen_title))
for index, score in sorted(title_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print(index)
    print("Score: {}\t Topic: {}".format(score, title_lda_model.print_topic(index, 5)))

"""And finally, on the test dataset."""

idx = 10
print("###################Processing#####################")
sample_nlp_pipeline(idx, test, test_title_bow_corpus)
print("###################Inference######################")
infer_topic_score(idx, test_title_bow_corpus)

"""### Initial conclusion of LDA results:
We can observe that our hypothesis that the resulting LDA model would encode the questions that are represented by tags into different topics seems to hold. Furthermore it seems to work on the test set as we observe ground truth tags in the topics.

## 4.4 Tags proposals from title topics

Now that we have a model capable of defining topics from titles, we can use the result of the inference to generate Tag predictions. Here we will use the topic confidence and each composant of the tokens of each topic. Since we are using an unsupervised approach we will extract tags using simply the confidence terms.

#### Proposed approach:
We will extract topic tags from a string of text and compare it with a set of available tokenized tags. If there is a match and it passes a confidence check, it is a tag suggestion and we obtain it using the tag2dict mapping. This allows us to not miss on predictions because the token is different than the tag.

First we need to preprcess the new sentence. Then, it is converted to the BOW representation used to train the LDA model. Inference is performed on this BOW set. We verfy that that using token2tag increases the probability of matches between suggestions and existing tags!
"""

# extract tag proposals from topics
unseen_title = 'How can I declare a variable in python'
bow_vector = title_dictionary.doc2bow(preprocess(unseen_title))

scores = []
words = []
for index, score in sorted(title_lda_model[bow_vector], key=lambda tup: -1*tup[1]):

    scores.append(score)
    words.append(title_lda_model.get_topic_terms(index, 5))

title_lda_model[bow_vector]

"""Check if there are matches with existing tags matches before applying token2tag"""

common_tags = []
# compare with dict of tags
for bow_id, score in words[0]:
  if title_dictionary[bow_id] in tags_array:
    common_tags.append(title_dictionary[bow_id])

print('common tags: {}'.format(common_tags))

"""Check if there are matches with existing tags after applying token2tag"""

common_tags = []
# compare with dict of tags
for bow_id, score in words[0]:
  token = title_dictionary[bow_id]
  # try to convert token2tag, else leave tag as is
  tag = ''
  if token in token2tag_dict.keys():
    tag = token2tag_dict[token]
  else:
    tag = token
  if tag in tags_array:
    common_tags.append(title_dictionary[bow_id])

print('common tags: {}'.format(common_tags))

"""Simply comparing for common tokens between the title and the set of tags would possibly work. However the LDA model approach allows for less computations and also provides a score that we can use to refine the prediction.

We have two scores per proposal. The overall confidence on the topic and then, for each topic, a score that represents token contribution to the topic.

Here we extract both scores of the topics and individual tokens to refine the tag proposal.
"""

# Extract topic score and individual token scores
title_token_score = []
title_topic_proposal = []
for index, topic_score in sorted(title_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    words = title_lda_model.get_topic_terms(index, 5)
    # compare with dict of tags
    for bow_id, score in words:
      if title_dictionary[bow_id] in tags_array:
        title_topic_proposal.append(title_dictionary[bow_id])
        title_token_score.append((topic_score, score))

"""Print the scores for each token of the title. We observe that in this case the individual token score is high, but the topic score is approximately the same for all found topics."""

for tag, score in zip(title_topic_proposal, title_token_score):
  print('tag : {}, topic score : {}, individual token score : {}'.format(tag, score[0], score[1]))

"""With the scores we now can threshold the result and obtain a refined tag suggestion:"""

# catch a tag given a threshold
title_tag_thresh = 0.1
for tag, score in zip(title_topic_proposal, title_token_score):
  if score[1] > title_tag_thresh:
    print('tag : {} ##### topic score : {} ##### individual score : {}'.format(tag, score[0], score[1]))

"""Since this is a suggestion system, we want to have high recall since giving a tag is better than no tag at all.

The final prediction list is the union between the confidence analysis and the tags present in the text.  If we use the intersection we will restrict the tags to the already existent tags. On the other hand using the individual token scores allows for the inclusion of new tags if they were already present before training!
"""

# Tag suggestion:
def lda_tag_suggestion(input_string, lda_model, corpus_dictionary,
                       token2tag_dict, n_proposals = 5, topic_thresh = 0.0,
                       token_thresh = 0.1, n_tags = 1000, new_tags = False, token2tag = True, verbose = True):
  """
  :input_string: string of text representing the title
  :lda_model: LDA model trained on corpus_dictionary
  :corpus_dictionary: Token dictionary
  :token2tag_dict: Token to tag encoding dictionary
  :n_proposals: maximum number of proposals
  :token2tag: set to False to disable token2tag
  :token2tag: set to True to accept new tags
  :title_topic_thresh: trim proposals based on topic score (default = 0)
  :title_token_thresh: trim proposals based on individual token score (default = 0.1)
  :verbose: silences function (default = False)

  :return: List of tag suggestions
  """
  # extract tag proposals from topics
  bow_vector = corpus_dictionary.doc2bow(preprocess(input_string))

  # Extract topic score and individual token scores
  token_scores = []
  proposals = []
  for index, topic_score in sorted(title_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    topic_composants = title_lda_model.get_topic_terms(index, 5)
    # check topic score
    if topic_score > topic_thresh:
      # enter topic composition
      for bow_id, token_score in topic_composants:
        # check token score
        if token_score > token_thresh:
          # compare with dict of tags
          token = corpus_dictionary[bow_id]
          tag = ''
          if token2tag:
            tag = token2tag_dict.get(token)

            if (tag == '' or tag == None) and new_tags:
              tag = token
          else:
            tag = token
          if tag != '' and tag != None:
            proposals.append(tag)
            token_scores.append(token_score)
            if verbose:
              print('tag : {} ##### topic score : {} ##### individual score : {}'.format(tag, topic_score, token_score))


  ordering = np.arange(len(proposals))
  proposals = np.array(proposals)
  token_scores = np.array(token_scores)
  # select top n_proposals by score:
  if len(proposals) > n_proposals:
    ordering = np.argsort(token_scores)
    proposals = proposals[ordering]
    token_scores = token_scores[ordering]
    return proposals[:n_proposals],token_scores[:n_proposals]
  else:
    return proposals, token_scores

"""Now we test the suggestion pipeline:

For sanity check we bserve that a string that is not working (possibly because tokenization did not work outputs no tag:
"""

test_str = ''
proposed_tags = lda_tag_suggestion(input_string = test_str, lda_model = title_lda_model, corpus_dictionary = title_dictionary,
                                   token2tag_dict = token2tag_dict, verbose = True)
print('tag proposals = {}'.format(proposed_tags))

"""Now we chec the result with a small string"""

test_str = 'Hello world'
proposed_tags,ps = lda_tag_suggestion(input_string = test_str, lda_model = title_lda_model, corpus_dictionary = title_dictionary,
                                   token2tag_dict = token2tag_dict, verbose = True)
print('tag proposals = {}'.format(proposed_tags))

"""Now with a possible title:

"""

test_str = 'How can i display keys and values of string python dict?'
proposed_tags, ps = lda_tag_suggestion(input_string = test_str, lda_model = title_lda_model, corpus_dictionary = title_dictionary,
                                   token2tag_dict = token2tag_dict, verbose = True)
print('tag proposals = {}'.format(proposed_tags))

"""Now we check what happens with a Title from the train dataset"""

# sample text
for _ in range(2):
  print('*******')
  idx = np.random.randint(len(df.Tags))
  title_text = df.Title.to_list()[idx]

  print('title: {}'.format(title_text))
  print('saple tags:', df.Tags.to_list()[idx])

  proposed_tags, _ = lda_tag_suggestion(input_string = title_text,  lda_model = title_lda_model, corpus_dictionary = title_dictionary,
                                   token2tag_dict = token2tag_dict, verbose = False)
  print('tag proposals = {}'.format(proposed_tags))

"""And now from the test dataset"""

# sample text
for _ in range(2):
  print('*******')
  idx = np.random.randint(len(test.Tags))
  title_text = test.Title.to_list()[idx]

  print('title: {}'.format(title_text))
  print('saple tags:', test.Tags.to_list()[idx])

  proposed_tags, _ = lda_tag_suggestion(input_string = title_text,  lda_model = title_lda_model, corpus_dictionary = title_dictionary,
                                   token2tag_dict = token2tag_dict, verbose = False)
  print('tag proposals = {}'.format(proposed_tags))

"""Print of processing pipeline + prediction + gt

## 4.5 Model evaluation

We consider for a tag $i$:

*   $TP_i$: The sum of the total number of correct predictions for the tag
*   $FP_i$: The sum of the total number of bad predictions for the tag
*   $TN_i$: All of the correctly non guessed tags are true negatives.
*   $FN_i$: We do not predict negatives such that this number is zero.

The first metric that would serve as an evaluation given the relevant parameters for the model the average F1-score for all tags.

However, since each tag is unequaly represented on the dataset (as seen on the exploratory analysis), we need to ponder this by the presence of each tag on the dataset.

The best suited evaluation metric for this problem is thus the micro-F1 score.

Micro F1-score (short for micro-averaged F1 score) is used to assess the quality of multi-label binary problems.
It measures the F1-score of the aggregated contributions of all classes.

It corresponds to pondering the average of each class prediction by it's appearence.


We define then

*   $TP = \sum_{\forall i} TP_i$
*   $FP = \sum_{\forall i} FN_i$
*   $N = $  Total number of tags

This quantities can be calculated without calculating the total ammount of

*   $Pr_{micro} = \frac{TP}{TP + FP}$
*   $Re_{micro}= \frac{TP}{N} = accuracy$

As a bonus, this formulation allows us to efficiently calculate the values without calculating $TP_i$ and $FP_i$ for each class!

And we use $F1_{micro} = 2\frac{Pr_{micro}*Re_{micro}}{Pr_{micro}+Re_{micro}}$ as our evaluation metric.
"""

# returns number of elements present in two lists
def count_matches(str_list_1, str_list_2):
  count = 0
  for word in str_list_1:
    if word in str_list_2:
      count+= 1
  return count

"""Now we check what happens with a Title from the train dataset."""

# calculate accuracy, recall, precision, TP, FP
total = 0
TP = 0
FP = 0
n_samples = len(df.Tags)//20
idxs = np.random.randint(len(df.Tags),size = n_samples)

for i, idx in enumerate(idxs):
  title_text = df.Title.to_list()[idx]
  gt_tags = df.Tags.to_list()[idx]
  proposed_tags, _ = lda_tag_suggestion(input_string = title_text,  lda_model = title_lda_model, corpus_dictionary = title_dictionary,
                                   token2tag_dict = token2tag_dict, verbose = False)
  gt_pos = len(gt_tags)
  pred_pos = len(proposed_tags)
  positives = count_matches(proposed_tags, gt_tags)
  total += gt_pos
  TP += positives
  # FP is given by the excedent of proposals.
  FP += max(pred_pos - TP, 0)
  if True:
    if i % int(n_samples/10) == 0:
      print('processed {} out of {} questions...'.format(i, n_samples))
acc = TP/total
precision = TP / (TP + FP)
recall = TP/(total)
f1 = precision*recall/(precision+recall)
print('Positives = {}'.format(total))
print('micro TP = {}'.format(TP))
print('micro FP = {}'.format(FP))
print('accuracy = {}'.format(acc))
print('micro precision = {}'.format(precision))
print('micro recall = {}'.format(recall))
print('micro f1 = {}'.format(f1))

"""And now from the test dataset"""

# calculate accuracy, recall, precision, TP, FP
total = 0
TP = 0
FP = 0
n_samples = len(test.Tags)//20
idxs = np.random.randint(len(test.Tags),size = n_samples)
for i, idx in enumerate(idxs):
  title_text = test.Title.to_list()[idx]
  gt_tags = test.Tags.to_list()[idx]
  proposed_tags, _ = lda_tag_suggestion(input_string = title_text,  lda_model = title_lda_model, corpus_dictionary = title_dictionary,
                                   token2tag_dict = token2tag_dict, verbose = False)
  gt_pos = len(gt_tags)
  pred_pos = len(proposed_tags)
  positives = count_matches(proposed_tags, gt_tags)
  total += gt_pos
  TP += positives
  # FP is given by the excedent of proposals.
  FP += max(pred_pos - TP, 0)
  if True:
    if i % int(n_samples/10) == 0:
      print('processed {} out of {} questions...'.format(i, n_samples))
acc = TP/total
precision = TP / (TP + FP)
recall = TP/(total)
f1 = precision*recall/(precision+recall)
print('Positives = {}'.format(total))
print('micro TP = {}'.format(TP))
print('micro FP = {}'.format(FP))
print('accuracy = {}'.format(acc))
print('micro precision = {}'.format(precision))
print('micro recall = {}'.format(recall))
print('micro f1 = {}'.format(f1))

"""# **4 Body Pre-processing**

Here we process the body of questions into a bow representation. Applying the same pipeline that we used for the Title is not straightforward since the text contains different sentences.
"""

df.Body.to_list()[:3]

processed_Bodies = df.Body.map(preprocess)

processed_Bodies[:10]

body_dictionary = gensim.corpora.Dictionary(processed_Bodies)
count = 0
for k, v in body_dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

body_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

body_bow_corpus = [body_dictionary.doc2bow(body) for body in processed_Bodies]

def body_sample_nlp_pipeline(sample_idx):
  print('sample idx:', sample_idx)

  print('saple tags:', df.Tags.to_list()[sample_idx])
  print('\nprocessing pipeline: \n')
  print('sample body:', df.Title.to_list()[sample_idx])
  print('preprocessed title:', processed_Bodies[sample_idx])
  print('bow_corpus of body:', body_bow_corpus[sample_idx])
  print('bag of words equivalence: \n')
  bow_doc_sample = body_bow_corpus[sample_idx]
  for i in range(len(bow_doc_sample)):
      print("Word {} (\"{}\") appears {} time.".format(bow_doc_sample[i][0],
                                                body_dictionary[bow_doc_sample[i][0]], bow_doc_sample[i][1]))

body_sample_nlp_pipeline(1)

"""### Question: How many topics should we use?"""

body_lda_model = gensim.models.LdaMulticore(body_bow_corpus, num_topics=20, id2word=body_dictionary, passes=20, workers=2)

# Save the model
model_file_path = "/content/drive/MyDrive/Colab Notebooks/Lda models/body_lda_model"
body_lda_model.save(model_file_path)

for idx, topic in body_lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

def infer_body_topic_score(sample_idx, ):
  for index, score in sorted(body_lda_model[body_bow_corpus[sample_idx]], key=lambda tup: -1*tup[1]):
      print("\nScore: {}\t \nTopic: {}".format(score, body_lda_model.print_topic(index, 10)))

idx = 11010
body_sample_nlp_pipeline(idx)
print("\n#####\nprediction:\n")
infer_body_topic_score(idx)

# test on unseen data
unseen_title = 'There seems to be an'
bow_vector = body_dictionary.doc2bow(preprocess(unseen_title))
for index, score in sorted(body_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {:.3f}\t Topcic: {}".format(score, body_lda_model.print_topic(index, 5)))

"""# Tag from body"""

# extract tag proposals from topics
unseen_body = 'How can I declare a struct in java if there is a list of random integers that I can lorem ipsum'
bow_vector = body_dictionary.doc2bow(preprocess(unseen_body))

scores = []
words = []
for index, score in sorted(body_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    scores.append(score)
    words.append(title_lda_model.get_topic_terms(index, 5))

words[0]

# compare with dict of tags
for bow_id, score in words[0]:
  if body_dictionary[bow_id] in tags_array:
    print(body_dictionary[bow_id])

body_tag_score = []
body_tag_proposal = []
for index, topic_score in sorted(body_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    words = body_lda_model.get_topic_terms(index, 5)
    # compare with dict of tags
    for bow_id, score in words:
      if body_dictionary[bow_id] in tags_array:
        body_tag_proposal.append(body_dictionary[bow_id])
        body_tag_score.append((topic_score, score))

for tag, score in zip(body_tag_proposal, body_tag_score):
  print('tag : {}, topic score : {}, individual score : {}'.format(tag, score[0], score[1]))

# catch a tag given a threshold
body_tag_thresh = 0.1
for tag, score in zip(body_tag_proposal, body_tag_score):
  if score[1] > body_tag_thresh:
    print('tag : {} ##### topic score : {} ##### individual score : {}'.format(tag, score[0], score[1]))

"""# 5. Title + body to tags:


"""

def infer_title_tags(title_text, thr = 0.1):
  # extract tag proposals from topics
  bow_vector = title_dictionary.doc2bow(preprocess(title_text))

  scores = []
  words = []
  for index, score in sorted(title_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
      scores.append(score)
      words.append(title_lda_model.get_topic_terms(index, 5))

  title_tag_score = []
  title_tag_proposal = []
  for index, topic_score in sorted(title_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
      words = title_lda_model.get_topic_terms(index, 5)
      # compare with dict of tags
      for bow_id, score in words:
        if title_dictionary[bow_id] in tags_array:
          title_tag_proposal.append(title_dictionary[bow_id])
          title_tag_score.append((topic_score, score))
  # catch a tag given a threshold
  proposals = {}
  for tag, score in zip(title_tag_proposal, title_tag_score):
    if score[1] > thr:
      #print('tag : {} ##### topic score : {} ##### individual score : {}'.format(tag, score[0], score[1]))
      proposals[tag] = (score[0], score[1])
  return proposals

def infer_body_tags(body_text, thr = 0.1):
  # extract tag proposals from topics
  bow_vector = body_dictionary.doc2bow(preprocess(body_text))

  scores = []
  words = []
  for index, score in sorted(body_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
      scores.append(score)
      words.append(body_lda_model.get_topic_terms(index, 5))

  body_tag_score = []
  body_tag_proposal = []
  for index, topic_score in sorted(body_lda_model[bow_vector], key=lambda tup: -1*tup[1]):
      words = body_lda_model.get_topic_terms(index, 5)
      # compare with dict of tags
      for bow_id, score in words:
        if body_dictionary[bow_id] in tags_array:
          body_tag_proposal.append(body_dictionary[bow_id])
          body_tag_score.append((topic_score, score))
  # catch a tag given a threshold
  proposals = {}
  for tag, score in zip(body_tag_proposal, body_tag_score):
    if score[1] > thr:
      #print('tag : {} ##### topic score : {} ##### individual score : {}'.format(tag, score[0], score[1]))
      proposals[tag] = (score[0], score[1])
  return proposals

infer_title_tags('i want to know how to create a python dict given a list of words')

infer_body_tags('i want to know how to create a python dict given a list of words')

import numpy as np

def infer_tag_proposals(title, body, thr=0.1, n_max=5):
    def aggregate_tags(tag_dict, proposals, scores):
        for tag, scores_tuple in tag_dict.items():
            if tag in proposals:
                idx = proposals.index(tag)
                scores[idx] += sum(scores_tuple)
            else:
                proposals.append(tag)
                scores.append(sum(scores_tuple))

    # Get title and body tag proposals
    title_tags = infer_title_tags(title, thr)
    body_tags = infer_body_tags(body, thr)

    # Aggregate tags and scores from title and body
    proposals = []
    scores = []

    aggregate_tags(title_tags, proposals, scores)
    aggregate_tags(body_tags, proposals, scores)

    # Sort proposals based on scores
    sorted_indices = np.argsort([-score for score in scores])
    sorted_proposals = [proposals[idx] for idx in sorted_indices]

    # Return only up to n_max proposals
    return sorted_proposals[:n_max]

# Example usage:
title = 'i want to know how to create a python dict given a list of words'
body = title
infer_tag_proposals(title, body, thr=0.1, n_max=5)

test1 = "i want to know how to create a python dict given a list of words"
infer_tag_proposals(test1, test1)

"""# Choosing an evaluation metric:

micro-averaged F1-score https://www.analyticsvidhya.com/blog/2020/12/hands-on-tutorial-on-stack-overflow-question-tagging/

# s
"""

pip install scikit-learn lightgbm gensim transformers tensorflow tensorflow-hub tabulate

from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

def create_nn_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

# separate tags into a list of tags using a lambda function
get_tags = lambda x: re.findall("\<(.*?)\>", x)

# Combine 'Title' and 'Body' for text representation
# df['Combined_Text'] = df['Title'] + ' ' + df['Body']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Title'], df['Tags'], test_size=0.2, random_state=42)

# Bag-of-Words representation
bow_vectorizer = CountVectorizer(max_features=5000, binary=True)
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)

# TF-IDF representation
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

y_test

# Combine the target labels from both train and test sets using concat
y_combined = pd.concat([y_train, y_test], ignore_index=True)

# Initialize MultiLabelBinarizer and fit-transform on the combined labels
mlb = MultiLabelBinarizer()
y_combined_binary = mlb.fit_transform(y_combined)

# Separate the binary labels back into train and test sets
y_train_binary = y_combined_binary[:len(y_train)]
y_test_binary = y_combined_binary[len(y_train):]

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize

# Assuming df is your DataFrame
tagged_data = [TaggedDocument(words=word_tokenize(str(_d).lower()), tags=[str(i)]) for i, _d in enumerate(df['Tags'])]
# Define Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=20, window=5, min_count=1, workers=4, epochs=20)
doc2vec_model.build_vocab(tagged_data)
doc2vec_model.train(tagged_data, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

# Function to obtain Doc2Vec embeddings
def get_doc2vec_embeddings(texts):
    return np.array([doc2vec_model.infer_vector(word_tokenize(str(text).lower())) for text in texts])



import torch
torch.cuda.get_device_name(0)

from tqdm.auto import tqdm
from transformers import BertTokenizer, BertModel
import torch

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Move BERT model to the GPU
bert_model = bert_model.to(device)

# Set batch size
batch_size = 16

# Tokenize and obtain BERT embeddings
def get_bert_embeddings(texts):
    # Tokenize texts in batches
    tokens_batch = [tokenizer(text, return_tensors='pt', truncation=True, padding=True) for text in texts]

    # Move tokens to the GPU
    tokens_batch = [{key: value.to(device) for key, value in tokens.items()} for tokens in tokens_batch]

    with torch.no_grad():
        # Get BERT embeddings in batches
        outputs_batch = [bert_model(**tokens) for tokens in tokens_batch]

    # Concatenate and average the last hidden states
    embeddings_batch = [outputs['last_hidden_state'].mean(dim=1).squeeze().cpu().numpy() for outputs in outputs_batch]

    return np.vstack(embeddings_batch)

# Apply BERT embeddings to train and test data in batches
X_train_bert = np.vstack([get_bert_embeddings(X_train[i:i+batch_size]) for i in tqdm(range(0, len(X_train), batch_size), desc="Processing Train Data")])
X_test_bert = np.vstack([get_bert_embeddings(X_test[i:i+batch_size]) for i in tqdm(range(0, len(X_test), batch_size), desc="Processing Test Data")])

import tensorflow as tf
import tensorflow_hub as hub

# Load Universal Sentence Encoder
use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function to obtain USE embeddings
def get_use_embeddings(texts):
    return np.array(use_embed(texts))

# Obtain USE embeddings
X_train_use = np.array(use_embed(X_train.tolist()))
X_test_use = np.array(use_embed(X_test.tolist()))
