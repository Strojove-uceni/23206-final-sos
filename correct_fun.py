import Levenshtein
import os
import torch.nn.functional as F

from model import CNNmodel
from data_module import TextRecognitionDataModule

from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
import torch


def download_and_unzip(url, extract_to='Datasets', chunk_size=1024*1024):
    http_response = urlopen(url)

    data = b''
    iterations = http_response.length // chunk_size + 1
    for _ in tqdm(range(iterations)):
        data += http_response.read(chunk_size)

    zipfile = ZipFile(BytesIO(data))
    zipfile.extractall(path=extract_to)

# Specify your local file path
# Specify your local file path
file_path = 'C:\\Users\\tagri\\Desktop\\IAM_Words'

# Check if the local file path exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The specified file path does not exist: {file_path}")

# Update the dataset_path to use the local file path
dataset_path = os.path.join(file_path, 'words')

# Optionally, you can check if the dataset_path exists and create it if needed
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Rest of your code remains the same
dataset, vocab, max_len = [], set(), 0

# Preprocess the dataset by the specific IAM_Words dataset file structure
words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[1] == "err":
        continue

    folder1 = line_split[0][:3]  # extract the folder name
    folder2 = "-".join(line_split[0].split("-")[:2])  # extract the sub-folder name
    file_name = line_split[0] + ".png"  # extract the file name
    label = line_split[-1].rstrip('\n')  # the label, last element of row, remove the \n at the end

    # the relative path to the image file
    rel_path = os.path.join(dataset_path, folder1, folder2, file_name)

    # append path and the label to dataset
    dataset.append([rel_path, label])

    # append label to vocab
    vocab.update(list(label))

    # length of the longest word
    max_len = max(max_len, len(label))

vocab = "".join(sorted(vocab))

data_module = TextRecognitionDataModule(dataset, vocab, max_len)
model = CNNmodel(pad_val=0, num_chars=len(vocab))
'''len(data_module.train_dataset.vocab)'''


def decode(output, blank_label=0):
    max_indices = torch.argmax(output, dim=-1)
    #print('max_indices:', max_indices)
    decoded_sequence = []
    for label in max_indices.cpu().numpy():
        if label != blank_label and (not decoded_sequence or label != decoded_sequence[-1]):
            decoded_sequence.append(label)
    print(decoded_sequence)

    return decoded_sequence

def label_to_string(label, vocab):
    strings = "".join([vocab[k] for k in label if k < len(vocab)])
    return strings

def calculate_accuracy(true_strings, pred_strings):
    correct_results = sum([1 for true, pred in zip(true_strings, pred_strings) if true == pred])
    total_results = len(true_strings)
    return correct_results, total_results

def calculate_digit_accuracy(true_strings, pred_strings, max_len):
    digit_correct = 0
    digit_total = 0

    for true, pred in zip(true_strings, pred_strings):
        for true_digit, pred_digit in zip(true, pred):
            digit_total += 1
            if true_digit == pred_digit:
                digit_correct += 1

    return digit_correct, digit_total

def correct(model, loader, vocab, blank_label=0):
    model.eval()
    device = next(model.parameters()).device

    #print('max_len:', data_module.max_len)

    correct_results = 0
    total_results = 0
    true_strings = []
    pred_strings = []
    digits_correct, digits_total = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            #print('y_hat:', y_hat)
            #print('y_shape:', y_hat.shape)

            decoded_preds = [decode(word) for word in y_hat]

            y_dec = []

            for label in y:
                decoded_sequence = []
                for i in label:
                    if i != blank_label and (not decoded_sequence or i != decoded_sequence[-1]):
                        decoded_sequence.append(i)
                y_dec.append(decoded_sequence)

            pred_strings.extend([label_to_string(pred, vocab) for pred in decoded_preds])
            true_strings.extend([label_to_string(target, vocab) for target in y_dec])

            cor_res, tot_res = calculate_accuracy(true_strings, pred_strings)
            correct_results += cor_res
            total_results += tot_res

            digit_correct, digit_total = calculate_digit_accuracy(true_strings, pred_strings, data_module.max_len)
            digits_correct.append(digit_correct)
            digits_total.append(digit_total)


    accuracy = correct_results / total_results if total_results > 0 else 0.0

    digit_correct = sum(digits_correct)
    digit_total = sum(digits_total)
    digit_accuracy = digit_correct / digit_total

    return accuracy, digit_accuracy, true_strings, pred_strings

train_accuracy, train_digit_accuracy, train_true_strings, train_pred_strings = correct(model, data_module.train_dataloader(), vocab)
print(f'Training set accuracy: {train_accuracy} correctly classified, an accuracy of {train_accuracy:.2f}%!')
print(f'Training set average CER: {train_digit_accuracy:.4f}')

val_accuracy, val_digit_accuracy, val_true_strings, val_pred_strings = correct(model, data_module.val_dataloader(), vocab)
print(f'Validation set accuracy: {val_accuracy} correctly classified, an accuracy of {val_accuracy:.2f}%!')
print(f'Validation set average CER: {val_digit_accuracy:.4f}')

for i in range(min(10, len(train_true_strings))):
    print(f'True: {train_true_strings[i]}, Predicted: {train_pred_strings[i]}')


