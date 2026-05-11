import os
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import DeepfakeDataset
from models.fusion_model import FusionModel


# =====================================================
# DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =====================================================
# DATASET
# =====================================================

dataset = DeepfakeDataset(
    csv_path="data/metadata.csv"
)

# remove failed samples
dataset = [x for x in dataset if x is not None]

print("Dataset size:", len(dataset))


# =====================================================
# TRAIN / VAL SPLIT
# =====================================================

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)


# =====================================================
# MODEL
# =====================================================

model = FusionModel().to(device)


# =====================================================
# LOSS + OPTIMIZER
# =====================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# =====================================================
# CHECKPOINTS
# =====================================================

os.makedirs("checkpoints", exist_ok=True)

best_val_acc = 0.0


# =====================================================
# TRAINING
# =====================================================

EPOCHS = 10

for epoch in range(EPOCHS):

    # =================================================
    # TRAIN
    # =================================================

    model.train()

    train_loss = 0
    train_preds = []
    train_labels = []

    for batch in train_loader:

        frames = batch["frames"].to(device)

        rppg = batch["rppg"].float().to(device)

        blink = batch["blink"].float().to(device)

        motion = batch["motion"].float().to(device)

        labels = batch["label"].float().to(device).squeeze(1)

        optimizer.zero_grad()

        outputs = model(
            frames,
            rppg,
            blink,
            motion
        ).squeeze(1)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        probs = torch.sigmoid(outputs)

        preds = probs > 0.5

        train_preds.extend(
            preds.cpu().numpy()
        )

        train_labels.extend(
            labels.cpu().numpy()
        )

    train_acc = accuracy_score(
        train_labels,
        train_preds
    )

    # =================================================
    # VALIDATION
    # =================================================

    model.eval()

    val_loss = 0
    val_preds = []
    val_probs = []
    val_labels = []

    with torch.no_grad():

        for batch in val_loader:

            frames = batch["frames"].to(device)

            rppg = batch["rppg"].float().to(device)

            blink = batch["blink"].float().to(device)

            motion = batch["motion"].float().to(device)

            labels = batch["label"].float().to(device).squeeze(1)

            outputs = model(
                frames,
                rppg,
                blink,
                motion
            ).squeeze(1)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.sigmoid(outputs)

            preds = probs > 0.5

            val_probs.extend(
                probs.cpu().numpy()
            )

            val_preds.extend(
                preds.cpu().numpy()
            )

            val_labels.extend(
                labels.cpu().numpy()
            )

    val_acc = accuracy_score(
        val_labels,
        val_preds
    )

    try:
        val_f1 = f1_score(
            val_labels,
            val_preds
        )
    except:
        val_f1 = 0.0

    try:
        val_auc = roc_auc_score(
            val_labels,
            val_probs
        )
    except:
        val_auc = 0.0

    # =================================================
    # SAVE BEST MODEL
    # =================================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            "checkpoints/best_model.pth"
        )

        print("Best model saved.")

    # =================================================
    # LOGGING
    # =================================================

    print("\n==========================")

    print(f"Epoch {epoch+1}/{EPOCHS}")

    print(f"Train Loss: {train_loss:.4f}")

    print(f"Train Accuracy: {train_acc:.4f}")

    print(f"Val Loss: {val_loss:.4f}")

    print(f"Val Accuracy: {val_acc:.4f}")

    print(f"Val F1: {val_f1:.4f}")

    print(f"Val AUC: {val_auc:.4f}")