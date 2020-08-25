import os
import matplotlib.pyplot as plt

from deepnp.dataloader import sequence
from deepnp.trainers import *


# from deepnp.layers import Attention


def eval_seq2seq(model, question, correct, id_to_char, verbose=False, is_reverse=False):
    correct = correct.flatten()
    # 머릿글자
    start_id = correct[0]
    correct = correct[1:]
    guess = model.generate(question, start_id, len(correct))

    # 문자열로 변환
    question = ''.join([id_to_char[int(c)] for c in question.flatten()])
    correct = ''.join([id_to_char[int(c)] for c in correct])
    guess = ''.join([id_to_char[int(c)] for c in guess])

    if verbose:
        if is_reverse:
            question = question[::-1]

        colors = {'ok': '\033[92m', 'fail': '\033[91m', 'close': '\033[0m'}
        print('Q', question)
        print('T', correct)

        is_windows = os.name == 'nt'

        if correct == guess:
            mark = colors['ok'] + '☑' + colors['close']
            if is_windows:
                mark = 'O'
            print(mark + ' ' + guess)
        else:
            mark = colors['fail'] + '☒' + colors['close']
            if is_windows:
                mark = 'X'
            print(mark + ' ' + guess)
        print('---')

    return 1 if guess == correct else 0


# 데이터셋 읽기
(x_train, y_train), (x_test, y_test) = sequence.load_data('date.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 개선: reverse 사용
# https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
do_reverse = True
if do_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]  # 각각 (N, T) 이기 때문

embedding_dim = 16
hidden_dim = 256
vocab_size = len(char_to_id)
batch_size = 256
n_epoch = 10
whole_batch = len(x_train)

model = Seq2SeqWithAttention(embedding_dim, hidden_dim, vocab_size)
optimizer = Adam
trainer = Trainer(model, optimizer, lr=0.001)
trainer.fit(x_train, y_train, n_epochs=1, batch_size=batch_size, verbose=2)


acc_list = []
for epoch in range(n_epoch):
    trainer.fit(x_train, y_train, n_epochs=1, batch_size=batch_size)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], y_test[[i]]
        verbose = i < 5
        if i % 1000 == 0: print(f"{round(i / len(x_test) * 100, 2)}%")
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse=True)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('정확도 %.3f%%' % (acc * 100))

print()
#
# # print()
# x = np.arange(len(acc_list))
# plt.plot(x, acc_list, marker='o')
# plt.xlabel('에폭')
# plt.ylabel('정확도')
# plt.ylim(0, 1.0)
# plt.show()
