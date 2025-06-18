import argparse
import glob
import os
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .swapper1024 import Swapper1024

class FaceDataset(Dataset):
    """Simple dataset returning target image and ArcFace embedding."""
    def __init__(self, root_dir):
        self.paths = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img_t = self.transform(img)
        embedding = torch.randn(512)  # placeholder embedding
        return img_t, embedding


def train(args):
    dataset = FaceDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = Swapper1024()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        for img, emb in loader:
            optimizer.zero_grad()
            out = model(img, emb)
            loss = nn.functional.l1_loss(out, img)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), 'swapper1024.pth')

    dummy_img = torch.randn(1, 3, 1024, 1024)
    dummy_emb = torch.randn(1, 512)
    torch.onnx.export(
        model, (dummy_img, dummy_emb), 'inswapper_1024.onnx',
        input_names=['img', 'embedding'],
        output_names=['output'],
        opset_version=11
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory with training images (1024x1024).')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    train(args)
