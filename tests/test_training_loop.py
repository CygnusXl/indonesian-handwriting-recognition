"""
Smoke test of the training loop on a tiny RANDOM dataset (seconds; runs on the
same device the training code uses, CPU or GPU).
Checks that the loop runs, writes history + a loadable checkpoint with
the configuration fields the app/evaluation rely on. The losses it
produces are meaningless.
"""

import csv

import torch
from torch.utils.data import DataLoader, Dataset

import htr_common as hc
import train
import train_baseline_extended as tbe
from dataloader import collate_fn


class RandomDataset(Dataset):
    def __init__(self, vocabulary, size=4):
        g = torch.Generator().manual_seed(0)
        self.items = []
        for _ in range(size):
            label = torch.randint(0, vocabulary.num_characters, (5,), generator=g)
            self.items.append(
                {
                    "image": torch.rand(1, 128, 1024, generator=g),
                    "label": label,
                    "text": vocabulary.decode(label.tolist()),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_train_loop_writes_history_and_checkpoint(tmp_path):
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-'\\")[:69]
    vocabulary = hc.CheckpointVocabulary({c: i for i, c in enumerate(chars)})

    dataset = RandomDataset(vocabulary)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # train_one_epoch moves every batch to train.DEVICE, so the model must be there
    # too (main() does the same); on a CPU-only machine this is a no-op.
    model = hc.build_model("cnn_bilstm", vocabulary.num_classes).to(train.DEVICE)

    checkpoint_path = str(tmp_path / "ckpt.pt")
    history_path = str(tmp_path / "history.csv")

    history, best = tbe.train_loop(
        model, loader, loader, vocabulary, 2, checkpoint_path, history_path
    )

    assert len(history["validation_cer"]) == 2

    with open(history_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["epoch"] for r in rows] == ["1", "2"]

    loaded, vocab, config, checkpoint = hc.load_model(checkpoint_path)
    assert config["image_height"] == 128 and config["ctc_timesteps"] == 256
    assert vocab.num_classes == 70
