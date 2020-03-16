import argparse
import os
import sys

sys.path.append(os.path.join(sys.path[0], 'training'))
sys.path.append(os.path.join(sys.path[0], 'prediction'))
from model_train import train
from model_predict import predict

if __name__ == '__main__':
    sys.path.append("path/foo/bar/")

    parser = argparse.ArgumentParser(
        usage=" -T to train ['--nocleaning & '--noidf'], -P to predict ['number_of_tags' 'url'], -h for help")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print debug info'
    )
    subparsers = parser.add_subparsers(dest='command')
    training = subparsers.add_parser('train', help="Trains the model, '-nc' skip cleaning & '-ni' skip IDF")
    training.add_argument('-nc', action='store_true', help="Skips the Data Cleaning Process")
    training.add_argument('-d', type=str, help="Enter the name of the dataset")
    training.add_argument('-t', type=int, help="Enter the rows to be truncated")

    prediction = subparsers.add_parser('predict', help="Predicts the output, 'main.py -P number_of_tags url'")
    prediction.add_argument('n', type=int)
    prediction.add_argument('url', type=str)

    args = parser.parse_args()
    if args.command == 'train':
        if args.t and args.d is not None:
            train(args.nc, args.d, args.t)
        elif args.t is None and args.d is None:
            train(args.nc, 'articles.csv', -1)
        elif args.t is None:
            train(args.nc, args.d, -1)
        elif args.d is None:
            train(args.nc, 'articles.csv', args.t)

    elif args.command == 'predict':
        predict(args.n, args.url)
