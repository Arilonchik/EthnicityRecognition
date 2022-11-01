import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2


class FairFaceDataset(Dataset):
    """Fair Face dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
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
        self.classes = list(self.labels["race"].unique())
        self.idx_to_class = {i: j for i, j in enumerate(classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}

    def __len__(self):
        return len(self.labels)

    def get_classes(self):
        return self.classes

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                os.path.basename(self.labels.iloc[idx, 0]))

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels.iloc[idx, 3]
        label = self.class_to_idx[label]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def gen_dataloader(data_folder, csv_file, transformers, batch_size):
    face_dataset = FairFaceDataset(csv_file=csv_file, root_dir=data_folder, transform=transformers)
    dataloader = DataLoader(face_dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4)
    return dataloader, len(face_dataset), face_dataset.get_classes()
