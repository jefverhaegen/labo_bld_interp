from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


IGNORE_IMAGES = [
    'c1f3f2cb1463bbfa905ccaff484cd668.png',  # truncated image
]


# Ignore 'DataFrame.swapaxes' is deprecated warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Ignore palette images with transparency expressed in bytes warnings
warnings.simplefilter(action='ignore', category=UserWarning)


class PokemonDataset(Dataset):
    def __init__(self, data_path, subset, k=5, val_fold=0, transform=None):
        data_path = Path(data_path)
        df = pd.DataFrame([
            {
                'image': str(img_path),
                'label': img_path.parent.name
            }
            for img_path in data_path.glob('data/*/*')
            if img_path.name not in IGNORE_IMAGES
        ])

        # Create mapping from label to integer
        label_to_int = {
            label: i
            for i, label in enumerate(sorted(df['label'].unique()))
        }

        # Split into train, test, val
        df_trainval, df_test = train_test_split(df, train_size=0.8,
                                                random_state=42)
        folds = np.array_split(df_trainval, k)
        df_val = folds[val_fold]
        train_folds = [fold for i, fold in enumerate(folds)
                       if i != val_fold]
        df_train = pd.concat(train_folds)

        # Store attributes
        self.data_path = data_path
        self.label_to_int = label_to_int
        self.transform = transform

        if subset == 'train':
            self.df = df_train.reset_index()
        elif subset == 'val':
            self.df = df_val.reset_index()
        elif subset == 'test':
            self.df = df_test.reset_index()
        else:
            raise ValueError(f'Unknown subset "{subset}"')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['image'][idx]
        label = self.df['label'][idx]
        int_label = self.label_to_int[label]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return (img, int_label)
