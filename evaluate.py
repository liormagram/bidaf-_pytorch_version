""" Customized version of the official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
import argparse
import json
import sys


def evaluate(dataset, predictions):
    total_labels = 0
    correct_labels = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total_labels += 1
                if str(qa['id']) not in predictions:
                    message = 'Unanswered question {} will receive score 0.'.format(str(qa['id']))
                    print(message, file=sys.stderr)
                    continue
                label = qa['answers'][0]['text']
                prediction = predictions[str(qa['id'])]
                if prediction == label:
                    correct_labels += 1

    accuracy = 100.0 * correct_labels / total_labels
    return accuracy


def main(args):
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        dataset = dataset_json['data']
    with open(args.dev_prediction_file) as dev_prediction_file:
        dev_predictions = json.load(dev_prediction_file)
    with open(args.test_prediction_file) as test_prediction_file:
        test_predictions = json.load(test_prediction_file)
    dev_accuracy = evaluate(dataset, dev_predictions)
    test_accuracy = evaluate(dataset, test_predictions)
    return dev_accuracy, test_accuracy


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    args = parser.parse_args()
    main(args)
