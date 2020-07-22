from typing import *
import nltk
from nlp_commons.functions import *
from nlp_commons.modules import *

from from_numpy.models import *

from nltk.corpus import stopwords
from string import punctuation

stopwords_agg = stopwords.words('english') + list(punctuation)
stopwords_hash = {k: 1 for k in stopwords_agg}


corpus = """경마장에 있는 말이 뛰고 있다\n
그의 말이 법이다\n
가는 말이 고와야 오는 말이 곱다\n"""

docs = [doc.strip() for doc in corpus.split('\n') if doc]
tokens_matrix = [[token for token in doc.split(' ') if token] for doc in docs]

uniques = get_uniques_from_nested_lists(tokens_matrix)
token2idx, idx2token = get_item2idx(uniques, unique=True, from_one=True)

sequences = []
for tokens in tokens_matrix:
    sequences += get_sequences_from_tokens(tokens, token2idx)

max_len = max([len(arr) for arr in sequences])
padded_sequences = pad_sequence_nested_lists(sequences, max_len, method='pre', truncating='pre')

seqs = np.array(padded_sequences)
X = seqs[:, :-1]
y = seqs[:, -1]

num_classes = len(token2idx) + 1  # 0번 인덱스는 one-hot에서 사용을 못함 (1 ~ 12 이므로)
y_onehot = to_categorical_iterable(y, num_classes)

# getting sample text
austen_fileids = [f for f in nltk.corpus.gutenberg.fileids() if f.startswith('austen')]
austen_fileids[0]
emma = nltk.corpus.gutenberg.words(austen_fileids[0])
# [token for token in text if not stopwords_hash.get(token)]
text = emma[:1000]
list(text)