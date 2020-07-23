from string import punctuation
import nltk.corpus

from deepnp.trainers import *
from nlp_commons.modules import *
from hidden_packages.langframe.functions import get_model_instance


def array_index_to_wv(arr, dim_restrict=None):
    vectors, used_tokens = [], []
    sample = wv.get_vector('hello')[:dim_restrict]
    try:
        for elem in arr:
            if elem == 0:
                vectors.append(np.zeros_like(sample))
            else:
                used_tokens.append(idx2token[elem])
                vectors.append(wv.get_vector(idx2token[elem])[:dim_restrict])
        return np.array(vectors), used_tokens
    except Exception as e:
        print(str(e))
        return 'No vector', _


def predict_rnn_lm(test_str):
    try:
        test_tokens = [s.strip() for s in test_str.split(" ") if not punkt.get(s)]
        indices = [token2idx[token] for token in test_tokens]
        x_test = pad_sequence_nested_lists([indices], max_len - 1, method='pre', truncating='pre')[0]
        x_test = np.array(x_test)
        x_test, _ = array_index_to_wv(x_test, dim_restrict=input_dim)
        pred_idx = model.predict(x_test)
        return idx2token[pred_idx]
    except Exception as e:
        return str(e)


stopwords_agg = stopwords.words('english') + list(punctuation)
stopwords_hash = {k: 1 for k in stopwords_agg}
punkt = {k: 1 for k in punctuation}

# loading word vector
wv = get_model_instance(model_name='glove_', model_filename='glove.6B.50d.txt', filepath='pretrained_models')

# getting sample text
austen_fileids = [f for f in nltk.corpus.gutenberg.fileids() if f.startswith('austen')]
emma = nltk.corpus.gutenberg.words(austen_fileids[0])
# stopwords 제거시 사용
# [token for token in text if not stopwords_hash.get(token)]

# token 정제
min_len, max_len = 3, 6
tokens = emma[:]
tokens = [token.strip() for token in tokens]

# 문장부호 기준으로 자르기
nested_tokens = get_tokens_sliced_by_punctuation(tokens, min_len=min_len, max_len=max_len)

# padding을 위해 token 길이 count
counts = [len(token_l) for token_l in nested_tokens]
min_len, max_len = min(counts), max(counts)

# token2idx, idx2token 생성
unique_tokens = get_uniques_from_nested_lists(nested_tokens)
token2idx, idx2token = get_item2idx(unique_tokens, unique=True, from_one=True)

# (0, 1), (0, 1, 2), (0, 1, 2, 3) 등으로 sequence 분할 후 padding
sequences = get_sequences_matrix(nested_tokens, token2idx)
padded_sequences = pad_sequence_nested_lists(sequences, max_len, method='pre', truncating='pre')

# input data와 정답 레이블 분리
seqs = np.array(padded_sequences)
X = seqs[:, :-1]
y = seqs[:, -1]

# one-hot으로 트레이닝 시킬 때 사용
# num_classes = len(unique_tokens) + 1  # 0번 인덱스는 one-hot에서 사용을 못함 (1 ~ 12 이므로)
# y_onehot = to_categorical_iterable(y, num_classes)

# Input Sample

sample_size = 500
input_dim = 50

X_vectors, total_used_tokens = [], []
for arr in X[:sample_size]:
    arr_wv, arr_used_tokens = array_index_to_wv(arr)
    if not arr_wv == 'No vector':
        X_vectors.append(arr_wv)
        total_used_tokens.append(arr_used_tokens)

X_vectors = np.array(X_vectors)
y_true = y[:len(X_vectors)]

print(X_vectors.shape)
print(y_true.shape)
print(len(total_used_tokens))

x_size = X_vectors.shape[0]
model = RNNTrainer(input_dim=input_dim, hidden_dim=200,
                   output_size=len(unique_tokens))
model.fit(X_vectors, y_true, batch_size=x_size, lr=0.5, n_epochs=1000, print_many=False)

predict_rnn_lm('miss')
print()

[' '.join(toks) for toks in total_used_tokens if len(toks) >= 5]