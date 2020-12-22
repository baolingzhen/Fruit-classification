import copy
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

from utils import check_paths, imshow
from dataloader import getDataLoader
from model import SimpleConvNet


def train(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    dl, params_ds = getDataLoader(args)
    train_dl, val_dl = dl
    train_size, val_size, classes_num = params_ds
    model = SimpleConvNet(classes_num, args.image_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    for epoch in tqdm(range(args.epochs)):
        train_loss = 0.0
        val_loss = 0.0
        accuracy = 0.0
        model.train()
        for inputs, labels in train_dl:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= train_size

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                accuracy += torch.sum(preds == labels.data).item()

        val_loss /= val_size
        accuracy /= val_size

        print(f"[epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={accuracy:.4f}]")

        if accuracy > best_acc:
            best_acc = accuracy
            best_model = copy.deepcopy(model.state_dict())

    torch.save(best_model, args.save_model_dir + f"/epochs{args.epochs}_best.pth")


def test(args):
    device = torch.device("cuda" if args.cuda else "cpu")
    test_ds, test_dl, classes_num = getDataLoader(args)
    model = SimpleConvNet(classes_num, args.image_size).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    accuracy = 0.0
    num_images = 9

    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            accuracy += torch.sum(preds == labels).item()

    for j in range(num_images):
        ax = plt.subplot(num_images // 3, 3, j + 1)
        ax.axis('off')
        ax.set_title(f'predicted: {test_ds.classes[preds[j]]}')
        imshow(images[j].cpu())
    plt.show()
    accuracy /= len(test_ds)
    print(f"Accuracy of the network on the test dataset is {100 * accuracy:4f}%")


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for neural network")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--epochs", type=int, default=10,
                                  help="number of training epochs, default is 10")
    train_arg_parser.add_argument("--batch-size", type=int, default=32,
                                  help="batch size for training, default is 32")
    train_arg_parser.add_argument("--dataset", type=str, required=True,
                                  help="path to training dataset, the path should point to a folder "
                                       "containing another folder with all the training images")
    train_arg_parser.add_argument("--save-model-dir", type=str, required=True,
                                  help="path to folder where trained model will be saved.")
    train_arg_parser.add_argument("--checkpoint-model-dir", type=str, default=None,
                                  help="path to folder where checkpoints of trained models will be saved")
    train_arg_parser.add_argument("--image-size", type=int, default=100,
                                  help="size of training images, default is 100 X 100")
    train_arg_parser.add_argument("--val-rate", type=float, default=0.2,
                                  help="the rate of training data used as validation set")
    train_arg_parser.add_argument('--cuda', action='store_true', help='enables cuda')
    train_arg_parser.add_argument("--seed", type=int, default=42,
                                  help="random seed for training")
    train_arg_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="learning rate, default is 1e-3")

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation arguments")
    eval_arg_parser.add_argument("--dataset", type=str, required=True,
                                 help="path for test dataset")
    eval_arg_parser.add_argument("--batch-size", type=int, default=64,
                                 help="batch size for training, default is 64")
    eval_arg_parser.add_argument("--image-size", type=int, default=100,
                                 help="size of training images, default is 100 X 100")
    eval_arg_parser.add_argument("--model", type=str, required=True,
                                 help="path to trained model, should be a exact path.")
    eval_arg_parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        torch.manual_seed(args.seed)
        check_paths(args)
        print(args)
        train(args)
        # train the model
    else:
        print(args)
        test(args)


if __name__ == '__main__':
    main()
