import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
from skimage import io
from PIL import Image


def make_class_to_index(csv_file):
    labels = pd.read_csv(csv_file)
    classes = list(labels["race"].unique())
    idx_to_class = {i: j for i, j in enumerate(classes)}
    class_to_idx = {value: key for key, value in idx_to_class.items()}
    return class_to_idx


class FairFaceDataset(Dataset):
    """Fair Face dataset."""

    def __init__(self, csv_file, root_dir, class_to_idx, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.labels)

    def get_classes(self):
        return self.class_to_idx

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                os.path.basename(self.labels.iloc[idx, 0]))

        # image = cv2.imread(img_name)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(img_name)
        #image = Image.fromarray(image)
        label = self.labels.iloc[idx, 3]
        label = self.class_to_idx[label]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def gen_dataloader(data_folder, csv_file, transformers, batch_size, class_to_idx):
    face_dataset = FairFaceDataset(csv_file=csv_file, root_dir=data_folder,
                                   transform=transformers, class_to_idx=class_to_idx)
    dataloader = DataLoader(face_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)
    return dataloader, len(face_dataset), face_dataset.get_classes()
