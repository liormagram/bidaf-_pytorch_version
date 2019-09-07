import argparse
import copy
import json
import os
from time import gmtime, strftime

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim

import evaluate
from model.data import SQuAD
from model.ema import EMA
from model.model import BiDAF


def train(args, data):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = BiDAF(args, data.WORD.vocab.vectors).to(device)
    best_model = copy.deepcopy(model)

    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir='runs/' + args.model_time)

    model.train()
    max_dev_accuracy = -1

    iterator = data.train_iter
    for epoch in range(args.epoch):
        num_iter = len(iterator)
        for i, batch in enumerate(iterator):

            b = model(batch)

            optimizer.zero_grad()
            batch_loss = criterion(b, batch.answer)
            loss = batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.update(name, param.data)

            if (i + 1) % args.print_freq == 0:
                dev_loss, accuracy = test(model, ema, args, data)
                c = (i + 1) // args.print_freq

                writer.add_scalar('loss/train', loss, c)
                writer.add_scalar('loss/dev', dev_loss, c)
                writer.add_scalar('accuracy/dev', accuracy, c)
                print('epoch: {}/{}, iteration: {}/{} train loss: {} / dev loss: {} / dev accuracy: {}'
                      .format(args.epoch, epoch, num_iter, i, rv(loss), rv(dev_loss), rv(accuracy)))

                if accuracy > max_dev_accuracy:
                    max_dev_accuracy = accuracy
                    best_model = copy.deepcopy(model)
                    torch.save(best_model.state_dict(), 'saved_models/BiDAF_{}.pt'.format(args.model_time))
                model.train()

    writer.close()
    print('max dev accuracy: {}'.format(max_dev_accuracy))

    return best_model


def test(model, ema, args, data):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    with torch.set_grad_enabled(False):
        for batch in iter(data.dev_iter):
            b = model(batch)
            batch_loss = criterion(b, batch.answer)
            loss += batch_loss.item()

            batch_size, _ = b.size()
            for i in range(batch_size):
                answer_id = batch.id[i]
                answer = torch.argmax(b[i]).item()
                answers[answer_id] = answer

        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup_params.get(name))

    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(answers), file=f)

    accuracy = evaluate.main(args)
    return loss, accuracy


# round value
def rv(value):
    return round(value, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='nlvr_dev.json')
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--epoch', default=12, type=int)
    parser.add_argument('--exp-decay-rate', default=0.999, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--hidden-size', default=100, type=int)
    parser.add_argument('--learning-rate', default=0.5, type=float)
    parser.add_argument('--print-freq', default=20, type=int)
    parser.add_argument('--train-batch-size', default=60, type=int)
    parser.add_argument('--train-file', default='nlvr_train.json')
    parser.add_argument('--word-dim', default=100, type=int)
    parser.add_argument('--max_c_len', default=20, type=int)
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'dataset_file', '.data/squad/{}'.format(args.dev_file))
    setattr(args, 'prediction_file', 'prediction{}.out'.format(args.gpu))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    print('data loading complete!')

    print('training start!')
    best_model = train(args, data)
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(best_model.state_dict(), 'saved_models/BiDAF_{}.pt'.format(args.model_time))
    print('training finished!')


if __name__ == '__main__':
    main()
