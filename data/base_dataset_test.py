from torch.utils.data import Dataset
import os
import glob
from torchvision import transforms
from PIL import Image


class BaseDatasetTest(Dataset):
    def __init__(self, path="./"):
        self.data = []
        self.labels = []
        self.path = path
        self._process_dataset(self.path)
        self.transform = self.get_transform()

    def _process_dataset(self, path):
        classes_dirs = glob.glob(os.path.join(path, "*/"))
        classes_dirs.sort()

        for i, classes in enumerate(classes_dirs):
            images = glob.glob(os.path.join(classes, "*.png"))
            labels = [i for _ in range(len(images))]
            self.data += images
            self.labels += labels

    def get_transform(self):

        return transforms.Compose(
            [
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, index):
        img = Image.open(self.data[index]).convert("RGB")
        label = self.labels[index]
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
