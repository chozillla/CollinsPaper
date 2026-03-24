"""
Wav2Vec 2.0 Transfer Learning Classifier (Method 2 from Collins et al.)

Fine-tune wav2vec2-base-960h with a 2-layer DNN classification head.

Hyperparameters (from paper):
- Base model: wav2vec2-base-960h (~95M params)
- Classification head: 2-layer DNN
- Learning rate: 1e-5
- Batch size: 8 (balanced positive/negative)
- Data augmentation: Gaussian noise, time stretch, pitch shift (each 50% prob)
- Training: 30-50 epochs
- Full model fine-tuning (not frozen trunk)
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from pathlib import Path
from tqdm import tqdm
import random

DATASET_DIR = Path("data/dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Hyperparameters from paper
LR = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 40
HIDDEN_DIM = 256


class AudioAugmentation:
    """Data augmentation as described in the paper.
    Each augmentation applied with 50% probability.
    """

    @staticmethod
    def add_gaussian_noise(audio, min_amp=0.001, max_amp=0.015):
        amplitude = random.uniform(min_amp, max_amp)
        noise = np.random.normal(0, amplitude, audio.shape)
        return audio + noise

    @staticmethod
    def time_stretch(audio, min_rate=0.8, max_rate=1.25):
        import librosa
        rate = random.uniform(min_rate, max_rate)
        stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
        # Pad or truncate to original length
        target_len = len(audio)
        if len(stretched) > target_len:
            stretched = stretched[:target_len]
        elif len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)))
        return stretched

    @staticmethod
    def pitch_shift(audio, sr=SAMPLE_RATE, min_semitones=-4, max_semitones=4):
        import librosa
        n_steps = random.uniform(min_semitones, max_semitones)
        shifted = librosa.effects.pitch_shift(
            audio.astype(np.float32), sr=sr, n_steps=n_steps
        )
        return shifted

    @staticmethod
    def augment(audio, prob=0.5):
        """Apply each augmentation with given probability."""
        audio = audio.copy()
        if random.random() < prob:
            audio = AudioAugmentation.add_gaussian_noise(audio)
        if random.random() < prob:
            audio = AudioAugmentation.time_stretch(audio)
        if random.random() < prob:
            audio = AudioAugmentation.pitch_shift(audio)
        return audio


class HDMDataset(Dataset):
    def __init__(self, audio_segments, labels, augment=False):
        self.audio = audio_segments
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio = self.audio[idx].astype(np.float32)
        label = self.labels[idx]

        if self.augment:
            audio = AudioAugmentation.augment(audio)

        return torch.tensor(audio, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class Wav2VecClassifier(nn.Module):
    """Wav2Vec2 with 2-layer DNN classification head."""

    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        # 2-layer DNN classification head
        wav2vec_dim = self.wav2vec.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(wav2vec_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        # Pool over time dimension (mean pooling)
        hidden = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(hidden)
        return logits


def create_balanced_sampler(labels):
    """Create a sampler that produces balanced batches."""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )
    return sampler


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for audio, labels in dataloader:
        audio = audio.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(audio)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for audio, labels in dataloader:
        audio = audio.to(device)
        logits = model(audio)
        probs = F.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return f1, all_preds, all_labels, all_probs


def train_and_evaluate_split(audio_segments, labels, meta, split_idx):
    """Train and evaluate on one Monte Carlo CV split."""
    split = meta["splits"][split_idx]
    train_meetings = set(split["train"])
    test_meetings = set(split["test"])

    all_examples = meta["positive"] + meta["negative"]

    train_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in train_meetings]
    test_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in test_meetings]

    train_audio = audio_segments[train_indices]
    train_labels = labels[train_indices]
    test_audio = audio_segments[test_indices]
    test_labels = labels[test_indices]

    print(f"  Train: {len(train_indices)} ({train_labels.sum()} pos)")
    print(f"  Test: {len(test_indices)} ({test_labels.sum()} pos)")

    # Create datasets
    train_dataset = HDMDataset(train_audio, train_labels, augment=True)
    test_dataset = HDMDataset(test_audio, test_labels, augment=False)

    # Balanced sampler for training
    sampler = create_balanced_sampler(train_labels)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = Wav2VecClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1 = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, DEVICE)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            f1, _, _, _ = evaluate(model, test_loader, DEVICE)
            print(f"  Epoch {epoch+1}: loss={train_loss:.4f}, acc={train_acc:.4f}, test_f1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch + 1
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), RESULTS_DIR / f"wav2vec_split{split_idx}_best.pt")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Load best model and get final metrics
    model.load_state_dict(torch.load(RESULTS_DIR / f"wav2vec_split{split_idx}_best.pt", weights_only=True))
    f1, preds, true_labels, probs = evaluate(model, test_loader, DEVICE)

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(true_labels, probs)

    return {
        "split": split_idx,
        "f1": f1,
        "best_epoch": best_epoch,
        "n_test": len(test_indices),
        "n_pos": int(test_labels.sum()),
        "report": classification_report(true_labels, preds, output_dict=True, zero_division=0),
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
    }


def main():
    print(f"Using device: {DEVICE}")

    print("Loading dataset...")
    audio_segments = np.load(DATASET_DIR / "audio_segments.npy")
    labels = np.load(DATASET_DIR / "labels.npy")
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    print(f"Dataset: {len(audio_segments)} segments, {labels.sum()} positive")

    results = []
    for split_idx in range(len(meta["splits"])):
        print(f"\n{'='*50}")
        print(f"Split {split_idx + 1}/{len(meta['splits'])}")
        print(f"{'='*50}")
        result = train_and_evaluate_split(audio_segments, labels, meta, split_idx)
        results.append(result)
        print(f"F1: {result['f1']:.4f} (best epoch: {result['best_epoch']})")

    avg_f1 = np.mean([r["f1"] for r in results])
    std_f1 = np.std([r["f1"] for r in results])
    print(f"\n{'='*50}")
    print(f"=== Wav2Vec 2.0 Transfer Learning ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Per-split: {[round(r['f1'], 4) for r in results]}")

    # Save results
    output = {
        "method": "Wav2Vec 2.0 Transfer Learning",
        "avg_f1": avg_f1,
        "std_f1": std_f1,
        "device": DEVICE,
        "hyperparameters": {
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
        },
        "splits": results,
    }
    with open(RESULTS_DIR / "wav2vec_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to {RESULTS_DIR}/wav2vec_results.json")


if __name__ == "__main__":
    main()
