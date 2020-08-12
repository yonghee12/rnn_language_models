from random import choice
from progress_timer import Timer

from deepnp.trainers import *
from deepnp.config import GPU

from hidden_packages.langframe.functions import *
from hidden_packages.langframe.functions import get_model_instance
from nlp_commons.modules import *

if GPU:
    if np.__name__ != 'cupy':
        import cupy as np
    np.cuda.Device(0).use()
    print("using gpu:", np.cuda.is_available())

import numpy as np

print()

stopwords_agg = stopwords.words('english') + list(punctuation)
stopwords_hash = {k: 1 for k in stopwords_agg}
punkt = {k: 1 for k in punctuation}


def predict_rnn_lm_bible(x_test, wv, idx2token, indexed=False):
    try:
        if not indexed:
            test_tokens = [s.strip() for s in x_test.split(" ") if not punkt.get(s)]
            indices = [token2idx[token] for token in test_tokens]
            x_test = np.array(indices)
        x_test, _ = array_index_to_wv_no_padding(x_test, wv, idx2token, dim_restrict=input_dim)
        pred_idx = model.predict(x_test)
        return idx2token[pred_idx]
    except Exception as e:
        return str(e)


def get_generated_sequence_bible(start_token):
    seq_str = start_token
    seq = [seq_str]
    for i in range(max_len - 2):
        seq.append(predict_rnn_lm_bible(seq_str))
        seq_str = ' '.join(seq)
    return seq_str


# loading word vector
wv = get_model_instance(model_name='glove_', model_filename='glove.6B.50d.txt', filepath='pretrained_models')
# wv = get_model_instance(model_name='glove_', model_filename='glove.840B.300d.txt', filepath='pretrained_models')

# 이 아래는 이 데이터 specific 전처리
with open('data/bible_in_basic_english.txt', 'r') as f:
    raw = f.readlines()
    f.close()


def get_selected_book_of_bible(bookname):
    return [line for line in raw if line.startswith(bookname)]


def get_bible_book_names(raw_bible_lines):
    book_names = {}
    for line in raw_bible_lines:
        local_book_name = line.split(' ')[0]
        if local_book_name.lower() == 'holy':
            continue
        if not book_names.get(local_book_name):
            book_names[local_book_name] = 1
    return list(book_names.keys())


# all_bibles = get_bible_book_names(raw)
all_bibles = ['Gen', 'Exo', 'Lev', 'Num', 'Deu', 'Jos', 'Jug', 'Rut', '1Sa', '2Sa', '1Ki', '2Ki', '1Ch', '2Ch', 'Ezr',
              'Neh', 'Est', 'Job', 'Psm', 'Pro', 'Ecc', 'Son', 'Isa', 'Jer', 'Lam', 'Eze', 'Dan', 'Hos', 'Joe', 'Amo',
              'Oba', 'Jon', 'Mic', 'Nah', 'Hab', 'Zep', 'Hag', 'Zec', 'Mal', 'Mat', 'Mak', 'Luk', 'Jhn', 'Act', 'Rom',
              '1Co', '2Co', 'Gal', 'Eph', 'Phl', 'Col', '1Ts', '2Ts', '1Ti', '2Ti', 'Tit', 'Phm', 'Heb', 'Jas', '1Pe',
              '2Pe', '1Jn', '2Jn', '3Jn', 'Jud', 'Rev']

books = ['Act', 'Rom', '1Co', '2Co', 'Gal', 'Eph', 'Phl', 'Col', '1Ts', '2Ts', '1Ti', '2Ti', 'Tit', 'Phm', 'Heb', 'Jas',
         '1Pe', '2Pe', '1Jn', '2Jn', '3Jn', 'Jud', ]

books = ['Act', 'Rom', '1Co', '2Co', 'Gal', 'Eph', 'Phl', 'Col']

texts = []
for book in books:
    texts += get_selected_book_of_bible(book)

# texts = get_selected_book_of_bible(book_of_bible)
# texts = texts[30:]
texts = process_multiple_lines_strip(texts)
tokenized_matrix = get_tokenized_matrix(texts, tokenizer='word_tokenize', exclude_stopwords=False,
                                        stopwords_options=[])

min_len, max_len = 3, 100
# 문장부호 기준으로 자르기
nested_tokens = []
for tokens in tokenized_matrix:
    nested_tokens += get_tokens_sliced_by_punctuation(tokens, min_len=min_len, max_len=max_len)

# token2idx, idx2token 생성
unique_tokens = get_uniques_from_nested_lists(nested_tokens)
token2idx, idx2token = get_item2idx(unique_tokens, unique=True, start_from_one=False)

# (0, 1), (0, 1, 2), (0, 1, 2, 3) 등으로 sequence 분할 후 padding
# bigram, trigram 등 사용
window_size = 3
sequences = get_sequences_matrix_window(nested_tokens, token2idx, window_size)

# input의 timestep이 서로 다를 경우 padding 필요
# padded_sequences = pad_sequence_nested_lists(sequences, max_len, method='pre', truncating='pre')

# padding 사용여부에 따라 다르게 설정
input_sequences = sequences
# input_sequences = padded_sequences

# input data와 정답 레이블 분리
seqs = np.array(input_sequences)
X = seqs[:, :-1]
y = seqs[:, -1]

# Input Sample

sample_size = None
input_dim = 50

X_vectors, total_used_tokens = [], []
for idx, arr in enumerate(X[:sample_size]):
    arr_wv, arr_used_tokens = array_index_to_wv_no_padding(arr, wv, idx2token)
    if not arr_wv == 'No vector':
        X_vectors.append(arr_wv)
        total_used_tokens.append(arr_used_tokens + [idx2token[y[idx]]])

X_vectors = np.array(X_vectors)
y_true = y[:len(X_vectors)]

print(X_vectors.shape)
print(y_true.shape)
print(len(total_used_tokens))

whole_batch = X_vectors.shape[0]
model = RNNTrainer(input_dim=input_dim, hidden_dim=500, output_size=len(unique_tokens), backend='numpy')
model.fit(X_vectors, y_true, batch_size=whole_batch//10, lr=0.1, n_epochs=30, print_many=False, verbose=1)

print()
# Generation logic
# test_sequences = get_sequences_from_tokens_window(unique_tokens, token2idx, window_size=2)
timer = Timer(len(total_used_tokens))
gens = []
for idx, test_sequence in enumerate(total_used_tokens):
    timer.time_check(idx)
    x_test = ' '.join(test_sequence[:-1])
    generated = predict_rnn_lm_bible(x_test, wv, idx2token, indexed=False)
    output_str = x_test + ' ' + generated
    gens.append(output_str)
    # print(output_str)

# Random generation logic
random_gens = []
for _ in range(1000):
    w1, w2 = choice(unique_tokens), choice(unique_tokens)
    w3 = predict_rnn_lm_bible(' '.join([w1, w2]), wv, idx2token, indexed=False)
    w4 = predict_rnn_lm_bible(' '.join([w2, w3]), wv, idx2token, indexed=False)
    output_str = ' '.join([w1, w2, w3, w4])
    random_gens.append(output_str)

loss_val = 'whole_matthew'
with open(f'results/bible_generated_w3_loss{loss_val}.txt', 'w') as f:
    f.write('\n'.join(gens))
    f.close()

with open(f'results/bible_random_generated_w3_loss{loss_val}.txt', 'w') as f:
    f.write('\n'.join(random_gens))
    f.close()

orgns = [' '.join(tokens) for tokens in total_used_tokens]
with open(f'results/bible_original_w3_loss{loss_val}.txt', 'w') as f:
    f.write('\n'.join(orgns))
    f.close()
