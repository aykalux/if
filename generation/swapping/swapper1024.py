import torch
import torch.nn as nn
import torch.nn.functional as F

class Swapper1024(nn.Module):
    """Simple U-Net style swapper for 1024x1024 images."""
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embed = nn.Linear(embedding_dim, embedding_dim)
        in_ch = 3 + embedding_dim
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, img: torch.Tensor, embedding: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        img: torch.Tensor
            Input target image tensor of shape (N, 3, 1024, 1024).
        embedding: torch.Tensor
            ArcFace embedding tensor of shape (N, 512).
        mask: torch.Tensor, optional
            Optional face mask tensor of shape (N, 1, 1024, 1024).
        """
        if mask is not None:
            img = torch.cat([img, mask], dim=1)
        e = self.embed(embedding).view(embedding.size(0), -1, 1, 1)
        e = e.expand(-1, e.size(1), img.size(2), img.size(3))
        x = torch.cat([img, e], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x
