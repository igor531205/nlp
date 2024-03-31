import argparse
import torch
import torch.nn.functional as F


def main():

    parser = argparse.ArgumentParser(
        description='Train RNN with data')

    parser.add_argument('model_torch', type=str,
                        help='torch file with train model')

    parser.add_argument('test_txt', type=str,
                        help='txt file with test data')

    args = parser.parse_args()

    # read file 'test.txt'
    link_test = args.test_txt
    words = open(link_test, 'r', encoding='utf-8').read().splitlines()

    # build the vocabulary of characters and mappings to/from integers
    chars = sorted(list(set(''.join(words))))
    stoi = {s: i+1 for i, s in enumerate(chars)}
    stoi['.'] = 0
    itos = {i: s for s, i in stoi.items()}
    num_chars = len(itos)

    # build the dataset
    block_size = 3  # context length: how many characters do we take to predict the next one?

    def build_dataset(words):
        X, Y = [], []
        for w in words:

            context = [0] * block_size
            for ch in w + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        return X, Y

    Xte, Yte = build_dataset(words)

    # Loading a Saved Model
    link_model = 'model.torch'
    model = torch.load(link_model)
    C, W1, b1, W2, b2 = model['C'], model['W1'], model['b1'], model['W2'], model['b2']

    # sample from the model
    g = torch.Generator().manual_seed(2147483647 + 10)

    emb = C[Xte]  # (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)  # (32, 100)
    logits = h @ W2 + b2  # (32, 27)
    loss = F.cross_entropy(logits, Yte)
    print(f'test loss: {loss}')

    for _ in range(5):

        out = []
        context = [0] * block_size  # initialize with all ...
        while True:
            emb = C[torch.tensor([context])]  # (1,block_size,d)
            h = torch.tanh(emb.view(1, -1) @ W1 + b1)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0:
                break

        print(''.join(itos[i] for i in out))


if __name__ == '__main__':
    main()
