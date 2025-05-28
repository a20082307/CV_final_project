import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import sklearn.metrics
import torch

from tqdm import tqdm
from CNN_dataloader import data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--new', action = 'store_true', help = 'Create a new model')
parser.add_argument('--name', type = str, default = 'cnn', help = 'The name of model')
parser.add_argument('--period', type = int, default = 16, help = 'The time period of the train and test data')
parser.add_argument('--interval', type = int, default = 1, help = 'The interval of the train and test data')
parser.add_argument('--which', type = int, default = 1, help = 'Choose the unit of kbar, 1 means 1hr, 3 means 15mins. 2 and 4 means find the difference between each two kbar')
parser.add_argument('--batch-size', type = int, default = 32, help = 'The batch size of the train data')
parser.add_argument('--epochs', type = int, default = 20, help = 'The number of epochs to train the model')
parser.add_argument('--lr', type = float, default = 0.001, help = 'The learning rate of the model')
args = parser.parse_args()

time_period = {
    1: '15min',
    2: '15min_diff',
    3: '1hr',
    4: '1hr_diff',
    5: '4hr',
    6: '4hr_diff'
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN(torch.nn.Module):
    def __init__(self, input_len, output_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels = 5, out_channels = 64, kernel_size = 5, padding = 2)
        self.conv2 = torch.nn.Conv1d(in_channels = 64, out_channels = 128, kernel_size = 5, padding = 2)
        self.pool = torch.nn.MaxPool1d(kernel_size = 2)
        self.relu = torch.nn.ReLU()

        dummy_input = torch.zeros(1, 5, input_len)
        x = self.pool(self.relu(self.conv1(dummy_input)))
        x = self.pool(self.relu(self.conv2(x)))
        flatten_dim = x.view(x.size(0), -1).shape[1]  # 得到展平後大小

        self.fc1 = torch.nn.Linear(flatten_dim, 256)
        self.fc2 = torch.nn.Linear(256, output_dim)

    def forward(self, x):  # x: (batch, 5, time_period)
        x = self.relu(self.conv1(x))  # (batch, 64, time_period)
        x = self.pool(x)              # (batch, 64, time_period // 2)
        x = self.relu(self.conv2(x))  # (batch, 128, time_period // 2)
        x = self.pool(x)              # (batch, 128, time_period // 4)
        x = x.view(x.size(0), -1)     # flatten
        # print(x.shape) 
        # x = self.relu(self.fc1(x))
        x = self.fc1(x)               # (batch, 256)
        x = self.relu(x)              # (batch, 256)
        x = self.fc2(x)
        return x

def train(model, train_data, train_label, val_data, val_label, criterion, optimizer):
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        start_pos = 0
        model.train()

        train_loss = 0
        for _ in tqdm(range(0, len(train_data) // args.batch_size + 1), desc = f'Epoch {epoch}:'):
            end_pos = start_pos + args.batch_size if start_pos + args.batch_size < len(train_data) else len(train_data)
            batch_data = train_data[start_pos : end_pos]
            batch_label = train_label[start_pos : end_pos]

            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * (end_pos - start_pos)
            start_pos = end_pos

        train_losses.append(train_loss / len(train_data))
        val_loss = validate(model, val_data, val_label, criterion)
        val_losses.append(val_loss)

    plt.plot(train_losses, label = 'Train Loss')
    plt.plot(val_losses, label = 'Validation Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {time_period[args.which]} data with {args.period} kbars in each data')
    plt.legend()
    plt.savefig(f'{args.name}_loss.png')
    plt.show()

def validate(model, val_data, val_label, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(val_data)), desc = 'Validating: '):
            input_data = val_data[i].unsqueeze(0)  
            label = val_label[i].unsqueeze(0)

            output = model(input_data)

            val_loss += criterion(output, label).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(val_label[i].view_as(pred)).sum().item()

    return val_loss / len(val_data)
    # accuracy = correct / len(val_data)
    # print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

def test(model, test_data, test_label, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    prediction = []
    with torch.no_grad():
        for i in tqdm(range(0, len(test_data)), desc = 'Validating: '):
            input_data = test_data[i].unsqueeze(0)  
            label = test_label[i].unsqueeze(0)

            output = model(input_data)

            val_loss += criterion(output, label).item()
            pred = output.argmax(dim = 1, keepdim = True)
            correct += pred.eq(test_label[i].view_as(pred)).sum().item()
            prediction.append(pred.item())

    val_loss /= len(test_data)
    accuracy = correct / len(test_data)
    print(f'Test Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    calculate_evaluation_metrics(test_label.cpu().numpy(), pred.cpu().numpy())

def calculate_evaluation_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics: precision, recall, and F1-score
    """
    # Get all true positives, false positives, and false negatives from confusion matrix
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()

    # Calculate global metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    cm_display = np.array([[tp, fp], [fn, tn]])
    print(f"Confusion Matrix:\n{cm_display}")
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_display, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Positive", "Negative"])
    plt.yticks(tick_marks, ["Positive", "Negative"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add text annotations in the cells
    thresh = (tp + tn) / 2
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm_display[i][j],
                horizontalalignment="center",
                color="white" if cm_display[i][j] > thresh else "black",
            )

    plt.tight_layout()
    plt.show()

    return precision, recall, f1_score

def main():
    train_data, train_label, test_data, test_label = data_loader(
        args.which,
        args.period,
        args.interval
    )
    print('=====' * 20)
    print(f'Current device: {device}')

    model_path = args.name + f'_which{args.which}_period{args.period}_interval{args.interval}_batch{args.batch_size}_epochs{args.epochs}_lr{args.lr}_' + '.pkl'
    model = None
    criterion = torch.nn.CrossEntropyLoss()
    if not os.path.exists(model_path) or args.new:
        print('=====' * 20)
        print('----- Model Structure -----')
        model = CNN(args.period, 2)
        print(model)
        print('=====' * 20)
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)

        val_data = train_data[int(len(train_data) * 0.8) : ]
        val_label = train_label[int(len(train_label) * 0.8) : ]
        train_data = train_data[: int(len(train_data) * 0.8)]
        train_label = train_label[: int(len(train_label) * 0.8)]
        print(f'Train data shape: {train_data.shape}, Train label shape: {train_label.shape}')
        print(f'Validation data shape: {val_data.shape}, Validation label shape: {val_label.shape}')
        print('=====' * 20)

        train_data_tensor = torch.tensor(train_data, dtype = torch.float32).to(device)
        train_label_tensor = torch.tensor(train_label, dtype = torch.long).to(device)
        val_data_tensor = torch.tensor(val_data, dtype = torch.float32).to(device)
        val_label_tensor = torch.tensor(val_label, dtype = torch.long).to(device)
        model = model.to(device)

        print('----- Training... -----')
        train(model, train_data_tensor, train_label_tensor, val_data_tensor, val_label_tensor, criterion, optimizer)
        print('=====' * 20)
        
        torch.save(model, model_path)
    else:
        model = torch.load(model_path, weights_only = False).to(device)
        print(f'Loading model from {model_path}')

    test_data_tensor = torch.tensor(test_data, dtype = torch.float32).to(device)
    test_label_tensor = torch.tensor(test_label, dtype = torch.long).to(device)
    print('----- Testing... -----')
    test(model, test_data_tensor, test_label_tensor, criterion)
    print('=====' * 20)


if __name__ == '__main__':
    main()