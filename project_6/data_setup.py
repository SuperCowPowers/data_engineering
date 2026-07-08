"""
data_setup.py - download Oxford-IIIT Pet, keep only the dog breeds we want, and
hand back ready-to-train DataLoaders. This is the plumbing; the modeling lives in
the classifier you'll write.

The first run downloads ~800 MB into project_6/data/ (git-ignored). Run the
scripts from inside project_6/.

Want more (or fewer) breeds? This BREEDS list is the ONLY knob — add or remove
names, then retrain. The wrapper below filters and re-labels automatically, and
the names are saved into the model, so Project 7 picks up the change for free.
Oxford-IIIT Pet ships 25 dog breeds total (the complete set is used below).
"""

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import OxfordIIITPet
from torchvision.models import ResNet18_Weights

# The dog breeds we'll classify (these exact names exist in Oxford-IIIT Pet).
BREEDS = [
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Boxer",
    "Chihuahua",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Miniature Pinscher",
    "Newfoundland",
    "Pomeranian",
    "Pug",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier",
]

# Every image gets the SAME preprocessing the pretrained ResNet expects
# (resize -> center-crop 224 -> normalize). Reusing its exact preprocessing is
# what lets us reuse its "eyes".
TRANSFORM = ResNet18_Weights.DEFAULT.transforms()


class _DogBreeds(Dataset):
    """Oxford-IIIT Pet filtered to our breeds, with labels remapped to 0..N-1.

    (We remap here rather than via torchvision's `target_transform`, because that
    is baked in at construction time and ignored if set afterward.)
    """

    def __init__(self, split):
        base = OxfordIIITPet(
            root="data",
            split=split,
            target_types="category",
            transform=TRANSFORM,
            download=True,
        )
        # each of our breeds' original label id -> a fresh 0..4 id
        self.remap = {base.class_to_idx[b]: i for i, b in enumerate(BREEDS)}
        self.base = base
        self.keep = [i for i, lbl in enumerate(base._labels) if lbl in self.remap]

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, i):
        image, label = self.base[self.keep[i]]  # base returns the raw 0-36 label
        return image, self.remap[label]  # we hand back 0..N-1


def train_loader(batch_size=32):
    return DataLoader(_DogBreeds("trainval"), batch_size=batch_size, shuffle=True, num_workers=0)


def test_loader(batch_size=32):
    return DataLoader(_DogBreeds("test"), batch_size=batch_size, shuffle=False, num_workers=0)
