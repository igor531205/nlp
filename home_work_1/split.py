import argparse
import random


def main():

    parser = argparse.ArgumentParser(
        description='Split on train and test data')

    parser.add_argument('txt_file', type=str,
                        help='txt file with data')

    args = parser.parse_args()

    # read file
    link = args.txt_file
    words = open(link, 'r', encoding='utf-8').read().splitlines()

    random.seed(42)
    random.shuffle(words)
    n1 = int(0.85 * len(words))  # 85% training data

    # Dividing data into training and test sets
    train_data = words[:n1]
    test_data = words[n1:]

    # Save to train.txt
    link_train = 'train.txt'
    with open(link_train, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write("%s\n" % item)

    # Save to test.txt
    link_test = 'test.txt'
    with open(link_test, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write("%s\n" % item)

    print(f'Names saved to {link_train} and {link_test}')


if __name__ == '__main__':
    main()
