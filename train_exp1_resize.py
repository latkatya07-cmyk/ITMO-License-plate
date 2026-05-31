import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

DATA_ROOT = Path(
    r"C:\Users\elatypova002\Desktop\ИТМО\2 семестр 1 курс\Проектирование архитектур нейронных систем\ДЗ 1\autoriaNumberplateOcrRu\mini_dataset"
)
IMG_HEIGHT = 64
IMG_WIDTH = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNBackbone(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
        )

    def forward(self, x):
        return self.net(x)


class CRNN(nn.Module):
    def __init__(self, img_h, num_channels, num_classes, hidden_size=256, num_layers=2):
        super().__init__()
        self.backbone = CNNBackbone(num_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=False,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        x = self.pool(feat)
        b, c, h, w = x.size()
        x = x.view(b, c, w)
        x = x.permute(2, 0, 1)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)


class CRNNDataset(Dataset):
    def __init__(self, root: Path, split: str, char_to_idx: dict, img_h: int, img_w: int):
        self.root = root
        self.split = split
        self.char_to_idx = char_to_idx
        self.idx_to_char = {v: k for k, v in char_to_idx.items()}
        self.img_h = img_h
        self.img_w = img_w
        self.samples = self._load_samples()
        print(f"[Dataset] {split}: {len(self.samples)} samples loaded")

    def _load_samples(self) -> List[Tuple[str, str]]:
        ann_dir = self.root / self.split / "ann"
        samples = []
        for jf in sorted(ann_dir.glob("*.json")):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            text = data.get("description", "").upper()
            img_name = jf.stem + ".png"
            img_path = self.root / self.split / "img" / img_name
            if img_path.exists():
                samples.append((str(img_path), text))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path_str, text = self.samples[idx]
        image = Image.open(img_path_str).convert("L")
        image = image.resize((self.img_w, self.img_h), Image.BICUBIC)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0)
        target = torch.tensor([self.char_to_idx.get(c, 0) for c in text], dtype=torch.long)
        return image, target, len(text)


def collate_fn(batch):
    images, targets, lengths = zip(*batch)
    images = torch.stack(images)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(lengths, dtype=torch.long)
    return images, targets, target_lengths


def load_alphabet(path: Path) -> Tuple[dict, dict]:
    with open(path, "r", encoding="utf-8") as f:
        chars = [line.strip() for line in f if line.strip()]
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_idx, idx_to_char


def decode_sequence(pred_ids: torch.Tensor, idx_to_char: dict, blank_idx: int) -> str:
    text = []
    prev = -1
    for idx in pred_ids.cpu().numpy():
        if idx != blank_idx and idx != prev:
            text.append(idx_to_char.get(idx, ""))
        prev = idx
    return "".join(text)


def compute_cer(pred: str, target: str) -> float:
    if not target:
        return float(len(pred) > 0)
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i-1] == target[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n] / n


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, targets, lengths in loader:
        images = images.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        output = model(images)
        input_lengths = torch.full((images.size(0),), output.size(0), device=device, dtype=torch.long)

        loss = criterion(output, targets, input_lengths, lengths)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device, blank_idx, idx_to_char):
    model.eval()
    total_loss, total_cer, correct = 0.0, 0.0, 0
    for images, targets, lengths in loader:
        images = images.to(device)
        targets = targets.to(device)
        lengths = lengths.to(device)

        output = model(images)
        input_lengths = torch.full((images.size(0),), output.size(0), device=device, dtype=torch.long)
        loss = criterion(output, targets, input_lengths, lengths)
        total_loss += loss.item()

        pred_ids = output.argmax(dim=2).transpose(0, 1)
        target_list = targets.split(lengths.tolist())

        for pred_row, tgt in zip(pred_ids, target_list):
            pred_text = decode_sequence(pred_row, idx_to_char, blank_idx)
            target_text = "".join([idx_to_char[int(t)] for t in tgt])
            total_cer += compute_cer(pred_text, target_text)
            if pred_text == target_text:
                correct += 1
    return total_loss / len(loader), total_cer / len(loader.dataset), correct / len(loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="baseline")
    return parser.parse_args()

def main():
    args = parse_args()
    
    char_to_idx, idx_to_char = load_alphabet(DATA_ROOT / "alphabet.txt")
    blank_idx = char_to_idx["<blank>"]
    num_classes = len(char_to_idx)

    train_ds = CRNNDataset(DATA_ROOT, "train", char_to_idx, IMG_HEIGHT, IMG_WIDTH)
    val_ds = CRNNDataset(DATA_ROOT, "val", char_to_idx, IMG_HEIGHT, IMG_WIDTH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = CRNN(IMG_HEIGHT, 1, num_classes).to(DEVICE)
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[{args.exp_name}] Device: {DEVICE} | Classes: {num_classes}")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_cer, val_acc = evaluate(model, val_loader, criterion, DEVICE, blank_idx, idx_to_char)
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | CER: {val_cer:.4f} | Acc: {val_acc:.4f}")

    save_path = DATA_ROOT / f"crnn_{args.exp_name}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()