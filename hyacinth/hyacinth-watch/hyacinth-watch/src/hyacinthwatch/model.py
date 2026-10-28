"""U-Net baseline for hyacinth segmentation (segmentation_models_pytorch).

This is a deliberately plain baseline. Once it works, compare against the
higher-level training utilities and foundation-model backbones in geoai
(check its docs for current function names before swapping in).
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ChipDataset(Dataset):
    def __init__(self, paths, mean, std, augment=False):
        self.paths, self.augment = list(paths), augment
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.std = np.asarray(std, np.float32)[:, None, None]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        d = np.load(self.paths[i])
        img = (d["image"] - self.mean) / self.std
        m = d["mask"].astype(np.float32)
        v = d["valid"].astype(np.float32)
        if self.augment:
            k = random.randint(0, 3)
            img, m, v = (np.rot90(a, k, axes=(-2, -1)) for a in (img, m, v))
            if random.random() < 0.5:
                img, m, v = (np.flip(a, axis=-1) for a in (img, m, v))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        return t(img), t(m)[None], t(v)[None]


def channel_stats(paths, max_chips=500):
    sel = list(paths)[:max_chips]
    arr = np.stack([np.load(p)["image"] for p in sel])
    return arr.mean(axis=(0, 2, 3)), arr.std(axis=(0, 2, 3)) + 1e-6


def build_model(in_channels, encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights=weights,
                    in_channels=in_channels, classes=1)


def masked_loss(logits, target, valid):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * valid).sum() / valid.sum().clamp(min=1)
    p = torch.sigmoid(logits) * valid
    t = target * valid
    dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
    return bce + dice


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    tp = fp = fn = 0.0
    for x, y, v in loader:
        x, y, v = x.to(device), y.to(device), v.to(device)
        pred = (torch.sigmoid(model(x)) >= thr).float() * v
        yt = y * v
        tp += (pred * yt).sum().item()
        fp += (pred * (1 - yt) * v).sum().item()
        fn += ((1 - pred) * yt).sum().item()
    return {
        "iou": tp / max(tp + fp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def balance(df, neg_ratio=2.0, seed=0):
    """Keep all chips with hyacinth and up to neg_ratio x as many empty ones."""
    pos = df[df.pos_frac > 0]
    neg = df[df.pos_frac == 0]
    neg = neg.sample(min(len(neg), int(neg_ratio * max(len(pos), 1))), random_state=seed)
    return df.loc[pos.index.union(neg.index)]


def train(index, mean, std, in_channels, encoder="resnet34", weights="imagenet",
          epochs=30, batch_size=16, lr=3e-4, neg_ratio=2.0, out_path="models/unet.pt",
          device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tr = balance(index[index.split == "train"], neg_ratio)
    va = index[index.split == "val"]
    dl_tr = DataLoader(ChipDataset(tr.path, mean, std, augment=True),
                       batch_size=batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ChipDataset(va.path, mean, std), batch_size=batch_size)
    model = build_model(in_channels, encoder, weights).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, history = -1.0, []
    for ep in range(epochs):
        model.train()
        total = 0.0
        for x, y, v in dl_tr:
            x, y, v = x.to(device), y.to(device), v.to(device)
            opt.zero_grad()
            loss = masked_loss(model(x), y, v)
            loss.backward()
            opt.step()
            total += loss.item()
        m = evaluate(model, dl_va, device)
        history.append({"epoch": ep, "loss": total / max(len(dl_tr), 1), **m})
        print(f"epoch {ep:02d} loss {history[-1]['loss']:.4f} val IoU {m['iou']:.3f} F1 {m['f1']:.3f}")
        if m["iou"] > best:
            best = m["iou"]
            torch.save(model.state_dict(), out_path)
    return model, history
