# ── Configuración de rutas ────────────────────────────────────────────────────
import sys, os
IN_COLAB = 'google.colab' in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/MR4010_proyecto_final_2026'
else:
    BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os
CSV_PATH   = os.path.join(BASE, 'data', 'dataset_merged.csv')
IMAGES_DIR = os.path.join(BASE, 'data', 'images')
MODEL_OUT  = os.path.join(BASE, 'models', 'cil_model_equipo25.pt')
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
print('BASE:', BASE)
print('CSV:', CSV_PATH)


# ── Imports ───────────────────────────────────────────────────────────────────
import csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter

# device: CUDA > MPS (Apple Silicon) > CPU
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print('Device:', DEVICE)


# ── Exploración del dataset ───────────────────────────────────────────────────
rows = list(csv.DictReader(open(CSV_PATH)))
cmds = Counter(int(r['nav_command']) for r in rows)
total = len(rows)
labels = {0:'CONTINUE', 1:'RECTO', 2:'IZQUIERDA', 3:'DERECHA'}
print(f'Total: {total:,}')
for c in range(4):
    n = cmds[c]
    bar = '█' * int(n/total*40)
    print(f'  {labels[c]:10s}: {n:6,} ({n/total*100:5.1f}%) {bar}')


# ── Dataset ───────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 200, 88    # resolución paper Codevilla 2017
N_COMMANDS   = 4

AUGMENT = transforms.Compose([
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.GaussianBlur(3, sigma=(0.1, 1.0)),
])

NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class CILDataset(Dataset):
    def __init__(self, rows, base_dir, augment=False):
        self.rows    = rows
        self.base    = base_dir
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        cmd   = int(row['nav_command'])
        steer = float(row['steering_angle'])
        speed = float(row['speed_kmh']) / 50.0   # normalizado 0-1

        img_path = os.path.join(self.base, row['image_path'])
        img = Image.open(img_path).convert('RGB').resize((IMG_W, IMG_H))

        # Flip horizontal: invierte steering y comando izq/der
        if self.augment and random.random() < 0.5:
            img   = img.transpose(Image.FLIP_LEFT_RIGHT)
            steer = -steer
            if cmd == 2: cmd = 3
            elif cmd == 3: cmd = 2

        if self.augment:
            img = AUGMENT(img)

        img_t   = NORMALIZE(img)
        speed_t = torch.tensor([speed], dtype=torch.float32)
        cmd_t   = torch.tensor(cmd, dtype=torch.long)
        steer_t = torch.tensor([steer], dtype=torch.float32)
        return img_t, speed_t, cmd_t, steer_t

# Split 85% train / 15% val estratificado por comando
random.seed(42)
by_cmd = {c: [r for r in rows if int(r['nav_command']) == c] for c in range(4)}
train_rows, val_rows = [], []
for c, rws in by_cmd.items():
    random.shuffle(rws)
    cut = int(len(rws) * 0.85)
    train_rows += rws[:cut]
    val_rows   += rws[cut:]

print(f'Train: {len(train_rows):,}  Val: {len(val_rows):,}')

train_ds = CILDataset(train_rows, BASE, augment=True)
val_ds   = CILDataset(val_rows,   BASE, augment=False)
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)


# ── Modelo CIL (Codevilla 2017) ───────────────────────────────────────────────
class CILModel(nn.Module):
    def __init__(self, n_commands=4):
        super().__init__()

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),  # 100x44
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 50x22
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                        # 25x11
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                           # 256x4x4=4096
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.2),
        )

        # Medición de velocidad
        self.speed_fc = nn.Sequential(
            nn.Linear(1, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )

        # Ramas por comando (Codevilla: una rama por acción)
        branch_input = 512 + 128
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(branch_input, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, 1), nn.Tanh(),   # salida: steering en [-1, 1]
            )
            for _ in range(n_commands)
        ])

    def forward(self, img, speed, cmd):
        feat  = self.cnn(img)
        spd   = self.speed_fc(speed)
        x     = torch.cat([feat, spd], dim=1)

        # Ejecutar todas las ramas, seleccionar la activa
        outs = torch.stack([b(x) for b in self.branches], dim=1)  # (B, 4, 1)
        idx  = cmd.view(-1, 1, 1).expand(-1, 1, 1)
        out  = outs.gather(1, idx).squeeze(1)                      # (B, 1)
        return out

model = CILModel(N_COMMANDS).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters())
print(f'Parámetros: {total_params:,}')


# ── Entrenamiento ─────────────────────────────────────────────────────────────
EPOCHS   = 40
LR       = 1e-4
MAX_STEER = 0.5   # el steering en el CSV está en radianes [-0.5, 0.5]

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

best_val_loss = float('inf')
history = {'train': [], 'val': []}

for epoch in range(1, EPOCHS + 1):
    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for img, speed, cmd, steer in train_dl:
        img   = img.to(DEVICE)
        speed = speed.to(DEVICE)
        cmd   = cmd.to(DEVICE)
        # normalizar steering a [-1, 1] para que Tanh cubra el rango
        steer = (steer / MAX_STEER).clamp(-1, 1).to(DEVICE)

        optimizer.zero_grad()
        pred = model(img, speed, cmd)
        loss = criterion(pred, steer)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dl)

    # ── Val ───────────────────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img, speed, cmd, steer in val_dl:
            img   = img.to(DEVICE)
            speed = speed.to(DEVICE)
            cmd   = cmd.to(DEVICE)
            steer = (steer / MAX_STEER).clamp(-1, 1).to(DEVICE)
            pred  = model(img, speed, cmd)
            val_loss += criterion(pred, steer).item()
    val_loss /= len(val_dl)

    history['train'].append(train_loss)
    history['val'].append(val_loss)
    scheduler.step()

    # guardar mejor modelo
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'n_commands': N_COMMANDS,
            'img_size': (IMG_W, IMG_H),
            'max_steer': MAX_STEER,
        }, MODEL_OUT)
        star = ' ★'
    else:
        star = ''

    print(f'Epoch {epoch:3d}/{EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}{star}')

print(f'\nMejor val_loss: {best_val_loss:.4f}')
print(f'Modelo guardado en: {MODEL_OUT}')


# ── Curvas de aprendizaje ─────────────────────────────────────────────────────
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(history['train'], label='Train MSE')
plt.plot(history['val'],   label='Val MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('CIL — Curva de aprendizaje')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE, 'models', 'learning_curve.png'), dpi=120)
plt.show()


# ── Verificación rápida del modelo guardado ───────────────────────────────────
ckpt = torch.load(MODEL_OUT, map_location='cpu')
print(f"Epoch guardado : {ckpt['epoch']}")
print(f"Val loss       : {ckpt['val_loss']:.5f}")
print(f"Img size       : {ckpt['img_size']}")
print(f"Comandos       : {ckpt['n_commands']}")

# inferencia de prueba
model_test = CILModel(ckpt['n_commands'])
model_test.load_state_dict(ckpt['model_state_dict'])
model_test.eval()
dummy_img   = torch.zeros(1, 3, IMG_H, IMG_W)
dummy_speed = torch.tensor([[0.5]])
dummy_cmd   = torch.tensor([0])
with torch.no_grad():
    pred = model_test(dummy_img, dummy_speed, dummy_cmd)
print(f"Inferencia OK  : steering={pred.item()*MAX_STEER:.4f} rad")
