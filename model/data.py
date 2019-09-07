import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, args):
        path = '.data/squad'
        dataset_path = path + '/torchtext/'
        train_examples_path = dataset_path + 'train_examples.pt'
        dev_examples_path = dataset_path + 'dev_examples.pt'
        test_examples_path = dataset_path + 'test_examples.pt'

        print("preprocessing data files...")
        if not os.path.exists('{}/{}l'.format(path, args.train_file)):
            self.preprocess_file('{}/{}'.format(path, args.train_file))
        if not os.path.exists('{}/{}l'.format(path, args.dev_file)):
            self.preprocess_file('{}/{}'.format(path, args.dev_file))
        if not os.path.exists('{}/{}l'.format(path, args.test_file)):
            self.preprocess_file('{}/{}'.format(path, args.test_file))

        self.RAW = data.RawField()
        # explicit declaration for torchtext compatibility
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'answer': ('answer', self.LABEL),
                       'context': ('c_emb', self.RAW),
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('answer', self.LABEL),
                       ('c_emb', self.RAW),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]

        if os.path.exists(dataset_path):
            print("loading splits...")
            train_examples = torch.load(train_examples_path)
            dev_examples = torch.load(dev_examples_path)
            test_examples = torch.load(test_examples_path)

            self.train = data.Dataset(examples=train_examples, fields=list_fields)
            self.dev = data.Dataset(examples=dev_examples, fields=list_fields)
            self.test = data.Dataset(examples=test_examples, fields=list_fields)
        else:
            print("building splits...")
            self.train, self.dev, self.test = data.TabularDataset.splits(
                path=path,
                train='{}l'.format(args.train_file),
                validation='{}l'.format(args.dev_file),
                test='{}l'.format(args.test_file),
                format='json',
                fields=dict_fields)

            os.makedirs(dataset_path)
            torch.save(self.train.examples, train_examples_path)
            torch.save(self.dev.examples, dev_examples_path)
            torch.save(self.test.examples, test_examples_path)

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev, self.test)
        self.WORD.build_vocab(self.train, self.dev, self.test, vectors=GloVe(name='6B', dim=args.word_dim))

        print("building iterators...")
        device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
        self.train_iter, self.dev_iter, self.test_iter = \
            data.BucketIterator.splits((self.train, self.dev, self.test),
                                       batch_sizes=[args.train_batch_size, args.dev_batch_size, args.test_batch_size],
                                       device=device,
                                       sort_key=lambda x: len(x.c_emb))

    def preprocess_file(self, path):
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(str(answer))

                            dictionary = dict([
                                ('id', id),
                                ('context', context),
                                ('question', question),
                                ('answer', answer),
                                ('s_idx', s_idx),
                                ('e_idx', e_idx)
                            ])
                            dump.append(dictionary)

        with open('{}l'.format(path), 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)
