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
    loss, last_epoch = 0, -1
    max_dev_accuracy = -1

    iterator = data.train_iter
    for epoch in range(args.epoch):
        if epoch % args.print_freq == 0:
            print('epoch: {} / {}'.format(epoch + 1, args.epoch))
        for i, batch in enumerate(iterator):
            # print('iteration: {}'.format(str(i)))

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
                dev_loss, test_loss, dev_accuracy, test_accuracy = test(model, ema, args, data)
                c = (i + 1) // args.print_freq

                writer.add_scalar('loss/train', loss, c)
                writer.add_scalar('loss/dev', dev_loss, c)
                writer.add_scalar('accuracy/dev', dev_accuracy, c)
                print('train loss: {} / dev loss: {} / dev accuracy: {} / test loss: {} / test accuracy: {}'.
                      format(loss, dev_loss, dev_accuracy, test_loss, test_accuracy))

                if dev_accuracy > max_dev_accuracy:
                    max_dev_accuracy = dev_accuracy
                    best_model = copy.deepcopy(model)
                    torch.save(best_model.state_dict(), 'saved_models/BiDAF_{}.pt'.format(args.model_time))

                model.train()

    writer.close()
    print('max dev accuracy: {}'.format(max_dev_accuracy))

    return best_model


def test(model, ema, args, data):
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.eval()

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    dev_iterator = data.dev_iter
    test_iterator = data.test_iter
    dev_loss, dev_answers = get_answers(backup_params, criterion, model, dev_iterator)
    test_loss, test_answers = get_answers(backup_params, criterion, model, test_iterator)

    with open(args.dev_prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(dev_answers), file=f)
    with open(args.test_prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(test_answers), file=f)

    dev_accuracy, test_accuracy = evaluate.main(args)
    return dev_loss, test_loss, dev_accuracy, test_accuracy


def get_answers(backup_params, criterion, model, iterator):
    answers = {}
    loss = 0
    with torch.set_grad_enabled(False):
        for batch in iter(iterator):
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
    return loss, answers

def test_best_model(best_weights_path, args, data):

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model = BiDAF(args, data.WORD.vocab.vectors).to(device)
    model.load_state_dict(torch.load(best_weights_path))
    model.eval()

    ema = EMA(args.exp_decay_rate)
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)

    backup_params = EMA(0)
    for name, param in model.named_parameters():
        if param.requires_grad:
            backup_params.register(name, param.data)
            param.data.copy_(ema.get(name))

    dev_iterator = data.dev_iter
    test_iterator = data.test_iter
    dev_loss, dev_answers = get_answers(backup_params, criterion, model, dev_iterator)
    test_loss, test_answers = get_answers(backup_params, criterion, model, test_iterator)

    with open(args.dev_prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(dev_answers), file=f)
    with open(args.test_prediction_file, 'w', encoding='utf-8') as f:
        print(json.dumps(test_answers), file=f)

    dev_accuracy, test_accuracy = evaluate.main(args)

    print('dev loss: {} / dev accuracy: {} / test loss: {} / test accuracy: {}'.
          format(dev_loss, dev_accuracy, test_loss, test_accuracy))

    return dev_loss, test_loss, dev_accuracy, test_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--char-dim', default=8, type=int)
    parser.add_argument('--char-channel-width', default=5, type=int)
    parser.add_argument('--char-channel-size', default=100, type=int)
    parser.add_argument('--context-threshold', default=400, type=int)
    parser.add_argument('--test-batch-size', default=100, type=int)
    parser.add_argument('--dev-batch-size', default=100, type=int)
    parser.add_argument('--dev-file', default='nlvr_dev.json')
    parser.add_argument('--test-file', default='nlvr_test.json')
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
    parser.add_argument('--max_c_len', default=600, type=int)
    args = parser.parse_args()

    print('loading SQuAD data...')
    data = SQuAD(args)
    setattr(args, 'char_vocab_size', len(data.CHAR.vocab))
    setattr(args, 'word_vocab_size', len(data.WORD.vocab))
    setattr(args, 'dataset_file', '.data/squad/{}'.format(args.dev_file))
    setattr(args, 'dev_prediction_file', 'dev_prediction{}.out'.format(args.gpu))
    setattr(args, 'test_prediction_file', 'test_prediction{}.out'.format(args.gpu))
    setattr(args, 'model_time', strftime('%H:%M:%S', gmtime()))
    print('data loading complete!')

    # print('training start!')
    # best_model = train(args, data)
    # if not os.path.exists('saved_models'):
    #     os.makedirs('saved_models')
    # torch.save(best_model.state_dict(), 'saved_models/BiDAF_{}.pt'.format(args.model_time))
    # print('training finished!')


    print('testing start')
    best_weights_path = 'BiDAF_16%3A37%3A43.pt'
    test_best_model(best_weights_path, args, data)
    print('testing finished')

if __name__ == '__main__':
    main()
