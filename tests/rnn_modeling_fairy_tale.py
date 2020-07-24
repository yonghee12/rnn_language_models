from random import choice

from deepnp.trainers import *
from hidden_packages.langframe.functions import get_model_instance
from nlp_commons.modules import *

stopwords_agg = stopwords.words('english') + list(punctuation)
stopwords_hash = {k: 1 for k in stopwords_agg}
punkt = {k: 1 for k in punctuation}

# loading word vector
wv = get_model_instance(model_name='glove_', model_filename='glove.6B.50d.txt', filepath='pretrained_models')


def get_separated_corpus(filepath):
    with open(filepath, 'r') as f:
        raw = f.readlines()
        f.close()

    title, author, body = list(), list(), list()
    flags = {'title': False, 'author': False, 'body': False}
    for line in raw:
        if line.startswith('<title>') or flags['title']:
            title.append(line)
            if line.startswith('<title>'):
                flags['title'] = True
            elif line.startswith('</title>'):
                flags['title'] = False
        elif line.startswith('<author>') or flags['author']:
            author.append(line)
            if line.startswith('<author>'):
                flags['author'] = True
            elif line.startswith('</author>'):
                flags['author'] = False
        else:
            body.append(line)

    return title, author, body


def get_word_from_merged_token(merged_token, token_option='lemmatized'):
    token_option = 'lemmatized'
    token_choices = {'original': 0, 'lemmatized': 1}
    idx = token_choices[token_option]
    token = merged_token.split(' --> ')[idx].strip()
    word, pos = token.split('/')
    return word


def predict_rnn_lm_tale(x_test, wv, idx2token, indexed=False):
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


filepaths = [
    'data/fairy_tales_corpus/cluster005/105115394-FATTY-AND-THE-GREEN-CORN-Arthur-Scott-Bailey.txt',
    'data/fairy_tales_corpus/cluster005/144629276-A-TERRIBLE-FRIGHT-Arthur-Scott-Bailey.txt',
    'data/fairy_tales_corpus/cluster005/228342269-FATTY-COON-AND-THE-MONSTER-Arthur-Scott-Bailey.txt',
    # 'data/fairy_tales_corpus/cluster005/471142238-JOHNNIE-GREEN-IS-DISAPPOINTED-Arthur-Scott-Bailey.txt',
    # 'data/fairy_tales_corpus/cluster005/522590877-FATTY-DISCOVERS-MRS.-TURTLE-S-SECRET-Arthur-Scott-Bailey.txt',
]

bodies = []
for path in filepaths:
    title, author, body = get_separated_corpus(path)
    bodies += body

# token punct로 context 분리
body_tokens = [get_word_from_merged_token(merged_token) for merged_token in bodies]
nested_tokens = get_tokens_sliced_by_punctuation(body_tokens, min_len=3, max_len=100)

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
    arr_wv, arr_used_tokens = array_index_to_wv_no_padding(arr, wv=wv, idx2token=idx2token, dim_restrict=None)
    if not arr_wv == 'No vector':
        X_vectors.append(arr_wv)
        total_used_tokens.append(arr_used_tokens + [idx2token[y[idx]]])

X_vectors = np.array(X_vectors)
y_true = y[:len(X_vectors)]

print(X_vectors.shape)
print(y_true.shape)
print(len(total_used_tokens))

whole_batch = X_vectors.shape[0]
model = RNNTrainer(input_dim=input_dim, hidden_dim=50, output_size=len(unique_tokens))
model.fit(X_vectors, y_true, batch_size=whole_batch, lr=0.85, n_epochs=1000, print_many=False)

# Generation logic
# test_sequences = get_sequences_from_tokens_window(unique_tokens, token2idx, window_size=2)
timer = Timer(len(total_used_tokens))
gens = []
for idx, test_sequence in enumerate(total_used_tokens):
    timer.time_check(idx)
    x_test = ' '.join(test_sequence[:-1])
    generated = predict_rnn_lm_tale(x_test, wv, idx2token, indexed=False)
    output_str = x_test + ' ' + generated
    gens.append(output_str)
    # print(output_str)

# Random generation logic
random_gens = []
for _ in range(300):
    w1, w2 = choice(unique_tokens), choice(unique_tokens)
    w3 = predict_rnn_lm_tale(' '.join([w1, w2]), wv, idx2token, indexed=False)
    w4 = predict_rnn_lm_tale(' '.join([w2, w3]), wv, idx2token, indexed=False)
    output_str = ' '.join([w1, w2, w3, w4])
    random_gens.append(output_str)

loss_val = '0.364'
with open(f'results/tale_size{whole_batch}_generated_w3_loss{loss_val}.txt', 'w') as f:
    f.write('\n'.join(gens))
    f.close()

with open(f'results/tale_size{whole_batch}_random_generated_w3_loss{loss_val}.txt', 'w') as f:
    f.write('\n'.join(random_gens))
    f.close()

orgns = [' '.join(tokens) for tokens in total_used_tokens]
with open(f'results/tale_size{whole_batch}_original_w3_loss{loss_val}.txt', 'w') as f:
    f.write('\n'.join(orgns))
    f.close()
