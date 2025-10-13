# scripts/train_weapon_classifier.py
import os, random
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----------------- synthetic data -----------------
def make_bg(h=64, w=64):
    img = np.random.normal(80, 10, (h, w)).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = (img.astype(np.float32) + np.linspace(0, 15, h)[:, None]).clip(0, 255).astype(np.uint8)
    return img

def draw_knife_mask(h=64, w=64):
    mask = np.zeros((h, w), np.uint8)
    tri = np.array([[10, 40], [40, 30], [40, 50]], np.int32)   # blade
    cv2.fillPoly(mask, [tri], 255)
    cv2.rectangle(mask, (40, 35), (54, 45), 255, -1)          # handle
    angle = random.uniform(-45, 45)
    scale = random.uniform(0.8, 1.3)
    M = cv2.getRotationMatrix2D((32, 32), angle, scale)
    mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
    tx, ty = random.randint(-6, 6), random.randint(-6, 6)
    M2 = np.array([[1, 0, tx],
               [0, 1, ty]], dtype=np.float32)
    mask = cv2.warpAffine(mask,M2, (w, h), flags=cv2.INTER_NEAREST)
    return mask

def synth_sample():
    bg = make_bg()
    y = 1 if random.random() < 0.5 else 0
    if y == 1:
        mask = draw_knife_mask()
        img = bg.copy()
        img[mask > 0] = (img[mask > 0] + random.randint(60, 120)).clip(0, 255)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    else:
        img = bg
    img = cv2.convertScaleAbs(img, alpha=1.0 + random.uniform(-0.2, 0.2), beta=random.uniform(-15, 15))
    return img, y

class SynthDataset(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx):
        img, y = synth_sample()
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0  # [1,64,64]
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

# ----------------- tiny CNN -----------------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64->32
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), # -> 1x1
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return torch.sigmoid(self.fc(x))  # [N,1]

def train():
    Path("models").mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(SynthDataset(12000), batch_size=128, shuffle=True)
    val_loader   = DataLoader(SynthDataset(2000),  batch_size=256)

    model = TinyCNN()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    best = 1e9
    for epoch in range(6):
        model.train()
        for x, y in train_loader:
            p = model(x)
            loss = loss_fn(p, y)
            opt.zero_grad(); loss.backward(); opt.step()

        # validation
        model.eval()
        losses = []
        with torch.no_grad():
            for x, y in val_loader:
                losses.append(loss_fn(model(x), y).item())
        val = sum(losses)/len(losses)
        print(f"epoch {epoch}: val_loss={val:.4f}")
        if val < best:
            best = val
            torch.save(model.state_dict(), "models/weapon_cnn.pt")

    # export ONNX compatible with the app
    model.load_state_dict(torch.load("models/weapon_cnn.pt", map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, 1, 64, 64)
    torch.onnx.export(
        model, 
        (dummy,), "models/model.onnx",
        input_names=["input"], output_names=["output"],
        opset_version=13, dynamic_axes=None
    )
    print("Saved → models/model.onnx")

if __name__ == "__main__":
    train()