import matplotlib.pyplot as plt
import pandas as pd

log_dir = 'version_10'


def plot_csv_losses(df_cleaned):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(df_cleaned['epoch'], df_cleaned['train_loss_epoch'], label='Training Loss', c='b')
    plt.plot(df_cleaned['epoch'], df_cleaned['val_loss_epoch'], label='Validation Loss', c='r')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


log_file = f'C:/Users/tagri/PycharmProjects/pythonProject/logs/TextRecognition/{log_dir}/metrics.csv'
df = pd.read_csv(log_file)
df_grouped = df.groupby('epoch').first().reset_index()
plot_csv_losses(df_grouped)