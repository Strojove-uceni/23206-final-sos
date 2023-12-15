import torch

import lightning.pytorch as pl

from PIL import Image


from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Resize, ColorJitter, RandomRotation, \
    RandomAdjustSharpness, RandomAutocontrast, AutoAugment



class IAMWordsDataset(Dataset):
    def __init__(self, dataset, vocab, max_len, split="train", train_val_split=0.9):
        self.dataset = dataset
        self.vocab = vocab
        self.max_len = max_len
        self.split = split
        self.train_val_split = train_val_split

        # maybe it should be assigned own vocab and max_len to train/val dataset, but idk
        if self.split == "train":
            self.dataset = self.dataset[:int(len(self.dataset) * self.train_val_split)]
        elif self.split == "val":
            self.dataset = self.dataset[int(len(self.dataset) * self.train_val_split):]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]
        image = Image.open(image_path).convert("RGB")
#------------------------------------------------------------------------------
        #print("Hello World")
        #torch.cuda.synchronize(print(f"Loaded image size: {image.size}"))
#------------------------------------------------------------------------------
        return image, label


class TextRecognitionDataModule(pl.LightningDataModule):
    def __init__(self, dataset, vocab, max_len, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.dataset = dataset
        self.vocab = vocab
        self.max_len = max_len

    def setup(self, stage=None):
        self.train_dataloader()
        self.val_dataloader()


    def train_dataloader(self):
        self.train_dataset = IAMWordsDataset(self.dataset, self.vocab, self.max_len, split="train")
        print("Train Dataloader called")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.train_collate, num_workers=0)

    def val_dataloader(self):
        self.val_dataset = IAMWordsDataset(self.dataset, self.vocab, self.max_len, split="val")
        print("Validation Dataloader called")
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          collate_fn=self.val_collate, num_workers=0)

    # ensure the same size of images and convert it to tensor
    def train_collate(self, batch):
        images, labels = zip(*batch)
        #print(labels)

        #--------------------------------------------------------------------
        # Print the shape of images before processing
        #for img in images:
            #print(f"Original image size: {img.size}")
        #--------------------------------------------------------------------

        # Resize all images to a consistent size (e.g., 128x32)
        resize = transforms.Compose([
            Resize((128, 32))])

        images = [resize(image) for image in images]

        transform = transforms.Compose([
            ColorJitter(brightness=0.5),
            RandomRotation(degrees=30),
            #RandomAffine(degrees=0, shear=10),
            #RandomAffine(degrees=0, translate=(0.1, 0.1)),
            #RandomAffine(degrees=0, scale=(0.8, 1.2)),
            RandomAdjustSharpness(sharpness_factor=2),
            RandomAutocontrast(),
            #RandomErasing(),
            AutoAugment(),
            ToTensor()])

        images = [transform(image) for image in images]

        # Convert labels to integer tensors

        converted_labels = []
        for label in labels:
            converted_label = [self.vocab.index(l) for l in label if l in self.vocab]
            converted_labels.append(torch.tensor(converted_label, dtype=torch.long))

        #print(converted_labels)

        # Pad the labels to the same length for batch processing
        padded_labels = torch.nn.utils.rnn.pad_sequence(converted_labels,
                                                        batch_first=True,
                                                        padding_value=0)
        return torch.stack(images), padded_labels

    def val_collate(self, batch):
        images, labels = zip(*batch)
        #print(labels)

        #--------------------------------------------------------------------
        # Print the shape of images before processing
        #for img in images:
            #print(f"Original image size: {img.size}")
        #--------------------------------------------------------------------

        # Resize all images to a consistent size (e.g., 128x32)
        transform = transforms.Compose([
            Resize((128, 32)),
            ToTensor()])

        images = [transform(image) for image in images]

        # Convert labels to integer tensors

        converted_labels = []
        for label in labels:
            converted_label = [self.vocab.index(l) for l in label if l in self.vocab]
            converted_labels.append(torch.tensor(converted_label, dtype=torch.long))

        #print(converted_labels)

        # Pad the labels to the same length for batch processing
        padded_labels = torch.nn.utils.rnn.pad_sequence(converted_labels,
                                                        batch_first=True,
                                                        padding_value=0)
        return torch.stack(images), padded_labels