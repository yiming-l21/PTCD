from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import csv
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer
from PIL import Image
from torchvision import transforms

@dataclass
class Sample:
    label: str
    img_id: str | None
    text_s: str
    text_a: str


class MSADataset(Dataset):

    def __init__(self, args, tsv_path: Path, dataset_name: str = 't2015'):
        self.tsv_file = tsv_path
        self.dataset = dataset_name
        self.no_img = args.no_img
        self.data_dir = Path(args.data_dir)
        self.img_dir = None if args.img_dir is None else Path(args.img_dir)
        self.lines = self._read_tsv(self.tsv_file)
        if not self.no_img:
            assert self.img_dir is not None
            self.img_dict = self._read_imgs()
            

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            lines.pop(0)  # remove the header row
            return lines
    def _read_imgs(self):
        img_dict = {}
        for line in self.lines:
            img_id = line[2]
            img = Image.open(os.path.join(self.img_dir, img_id)).convert('RGB')
            fname = self.tsv_file.name if isinstance(self.tsv_file, Path) else os.path.basename(self.tsv_file)
            if 'train' in fname:
                img = transforms.Resize([256, 256])(img)
                img = transforms.RandomCrop([224, 224])(img)
            else:
                img = transforms.Resize([224, 224])(img)
            img = transforms.ToTensor()(img)  # (3, 224, 224)
            img_dict[img_id] = img
        return img_dict
    def __len__(self):
        return len(self.lines)

    def read(self) -> List[Sample]:
        samples: List[Sample] = []
        for line in self.lines:
            label = line[1].strip() if len(line) > 1 else ""
            img_id = line[2].strip() if len(line) > 2 and line[2].strip() != "" else None
            text_s = (line[3] if len(line) > 3 else "").lower()
            text_a = (line[4] if len(line) > 4 else "").lower()
            # special substitution for Twitter datasets
            if self.dataset in ['t2015', 't2017']:
                text_s = text_s.replace('$t$', text_a)

            samples.append(Sample(label=label, img_id=img_id, text_s=text_s, text_a=text_a))
        return samples
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        label_id = self.label_id_map[line[1]]
        img_id = line[2]
        text_s = line[3].lower()
        text_a = line[4].lower()
        # special substitution for Twitter datasets
        if self.dataset in ['t2015', 't2017']:
            text_s = text_s.replace('$t$', text_a)

        if self.no_img:
            img = None
        else:
            img = self.img_dict[img_id]
            
        return {
            # img
            'img_id': img_id,
            'img': img,
            # text
            "text_s": text_s,
            "text_a": text_a
        }