import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np

from utils.tokenizer import LaneTokenizer


class TuSimpleDataset(Dataset):
    def __init__(self, root_dir, split='train', nbins=1000, format_type='anchor', image_size=(320, 800)):
        self.root_dir = root_dir
        self.split = split
        self.nbins = nbins
        self.format_type = format_type
        self.image_size = image_size  # (height, width)
        self.tokenizer = LaneTokenizer(nbins=self.nbins)

        self.samples = []

        # ✅ Prepare label files
        if split == 'train':
            label_files = ['label_data_0313.json', 'label_data_0531.json', 'label_data_0601.json']
        elif split == 'test':
            label_files = ['test_label.json']
        else:
            raise ValueError(f"Unsupported split: {split}")

        # ✅ Read all samples
        for label_file in label_files:
            label_path = os.path.join(root_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))

        # ✅ Define augmentations
        self.transform = self.get_transforms(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.root_dir, sample['raw_file'])
        image = Image.open(image_path).convert('RGB')

        # ✅ Save original PIL size (before resizing), which is (width, height)
        original_pil_size = image.size
        #print(f"[DEBUG] Loaded original PIL image size (width, height): {original_pil_size}")

        # ✅ Resize PIL image to target size: (width, height)
        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)

        # ✅ After resize, image.size is (width, height)
        resized_pil_size = image.size
        #print(f"[DEBUG] After resizing, PIL image size (width, height): {resized_pil_size}")

        # ✅ Apply augmentations (now image becomes Tensor)
        image = self.transform(image)

        # ✅ After transform, image is tensor: (C, H, W)
        _, height, width = image.shape
        #print(f"[DEBUG] After transforms, tensor shape: {image.shape}")
        #print(f"[DEBUG] Transformed image size passed to tokenizer (width, height): ({width}, {height})")

        # ✅ Prepare annotation: pass (width, height) of transformed image
        #annotation = self._convert_annotation(sample, original_size=(width, height))
        #print(f"[DEBUG] Annotation prepared with image size: (width, height) = ({width}, {height})")
        original_pil_width, original_pil_height = original_pil_size  # already extracted before
        annotation = self._convert_annotation(
            sample,
            original_size=(original_pil_width, original_pil_height),
            target_size=(width, height)
        )

        # ✅ Tokenize
        input_seq, target_seq = self.tokenizer.encode(
            annotation,
            (width, height),
            format_type=self.format_type
        )

        return {
            'image': image,
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'target_seq': torch.tensor(target_seq, dtype=torch.long),
            'raw_file': sample['raw_file']
        }

    def get_transforms(self, split):
        if split == 'train':
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    shear=5,
                    interpolation=T.InterpolationMode.BILINEAR
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

    def _convert_annotation(self, sample, original_size, target_size):
        lanes = []
        h_samples = sample['h_samples']
        img_width, img_height = original_size
        new_width, new_height = target_size
    
        x_scale = new_width / img_width
        y_scale = new_height / img_height
    
        for lane_points in sample['lanes']:
            points = []
            for x, y in zip(lane_points, h_samples):
                if x != -2:
                    # ✅ Scale points according to resized image
                    new_x = x * x_scale
                    new_y = y * y_scale
                    points.append([new_x, new_y])
    
            if points:
                if self.format_type == 'parameter':
                    points_np = np.array(points)
                    xs = points_np[:, 0]
                    ys = points_np[:, 1]
    
                    if len(xs) >= 5:
                        ys_min, ys_max = ys.min(), ys.max()
                        ys_norm = (ys - ys_min) / (ys_max - ys_min + 1e-8)
                        xs_norm = xs / new_width
    
                        coeffs = np.polyfit(ys_norm.astype(np.float32), xs_norm.astype(np.float32), deg=4)
    
                        lanes.append({
                            'params': coeffs.tolist(),
                            'offset': float(ys_min),
                            'ys_max': float(ys_max)
                        })
                else:
                    lanes.append({'points': points})
    
        if not lanes:
            print(f"[WARNING] No valid lanes found in sample: {sample['raw_file']}")
            return {'lanes': []}
    
        return {'lanes': lanes}

