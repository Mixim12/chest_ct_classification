from torch.utils.data import Dataset
import os
import glob
from torchvision import transforms
from PIL import Image


class BaseDataset(Dataset):
    def __init__(self, path="./", mode="train"):
        self.data = []
        self.labels = []
        self.path = path
        self.mode = mode
        self._process_dataset(self.path, self.mode)
        self.transform = self.get_transform(mode)

    def _process_dataset(self, path, mode="train"):
        classes_dirs = glob.glob(os.path.join(path, "*/"))
        classes_dirs.sort()

        for i, classes in enumerate(classes_dirs):
            images = glob.glob(os.path.join(classes, "*.png"))
            labels = [i for _ in range(len(images))]
            self.data += images
            self.labels += labels

    def get_transform(self, mode):
        if mode == "train":
            return transforms.Compose(
                [
                    transforms.Resize((500, 500)),
                    transforms.RandomCrop((400, 400)),
                    transforms.RandomHorizontalFlip(p=0.3),
                    transforms.ColorJitter(),
                    transforms.ToTensor(),
                ]
            )
        elif mode == "valid":
            return transforms.Compose(
                [transforms.Resize((400, 400)), transforms.ToTensor()]
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert("RGB")
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
