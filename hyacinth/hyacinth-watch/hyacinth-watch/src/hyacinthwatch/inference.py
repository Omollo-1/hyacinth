"""Sliding-window inference over full scenes and raster export."""
import numpy as np
import torch


@torch.no_grad()
def predict_scene(model, image, mean, std, device, tile=512, overlap=64):
    """image: (C, H, W) reflectance -> (H, W) hyacinth probability."""
    _, H, W = image.shape
    x = np.nan_to_num(image)
    x = (x - np.asarray(mean, np.float32)[:, None, None]) / np.asarray(std, np.float32)[:, None, None]
    prob = np.zeros((H, W), np.float32)
    wts = np.zeros((H, W), np.float32)
    step = tile - overlap
    seen = set()
    model.eval()
    for r in range(0, H, step):
        for c in range(0, W, step):
            r0, c0 = min(r, max(H - tile, 0)), min(c, max(W - tile, 0))
            if (r0, c0) in seen:
                continue
            seen.add((r0, c0))
            patch = x[:, r0:r0 + tile, c0:c0 + tile]
            ph, pw = patch.shape[1:]
            pad = ((0, 0), (0, (-ph) % 32), (0, (-pw) % 32))
            t = torch.from_numpy(np.pad(patch, pad, mode="edge")).float()[None].to(device)
            p = torch.sigmoid(model(t))[0, 0, :ph, :pw].cpu().numpy()
            prob[r0:r0 + ph, c0:c0 + pw] += p
            wts[r0:r0 + ph, c0:c0 + pw] += 1
    return prob / np.maximum(wts, 1)


def save_raster(array, like, path, dtype="float32"):
    """Write a 2D array to GeoTIFF using the grid of DataArray `like` (band, y, x)."""
    ref = like.isel(band=0, drop=True) if "band" in like.dims else like
    out = ref.copy(data=np.asarray(array).astype(dtype))
    out.rio.to_raster(path, compress="deflate")
