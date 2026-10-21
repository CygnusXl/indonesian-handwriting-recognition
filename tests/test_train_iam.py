"""IAM training loop on fake data: wiring only; the losses mean nothing."""

import csv
import random

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

import htr_common as hc
import iam_dataset as iam
import train
import train_iam
from dataloader import collate_fn


def fake_pairs(texts):
    pairs = []
    for text in texts:
        image = Image.new("L", (200, 40), 255)
        ImageDraw.Draw(image).text((8, 10), text, fill=0)
        pairs.append((image, text))
    return pairs


def test_train_loop_saves_loadable_checkpoint_with_provenance(tmp_path):
    texts = ["abc", "bca", "cab", "abca"]
    vocabulary = iam.build_vocabulary(texts)
    dataset = iam.IamLineDataset.from_pairs(
        fake_pairs(texts), vocabulary, image_width=256, ctc_timesteps=64,
        augment=True, rng=random.Random(0),
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = hc.build_model("cnn_bilstm", vocabulary.num_classes).to(
        train.DEVICE
    )

    checkpoint_path = str(tmp_path / "iam.pt")
    history_path = str(tmp_path / "history.csv")

    history, best = train_iam.train_loop(
        model, loader, loader, vocabulary, 2, checkpoint_path, history_path,
        image_width=256,
        extra_checkpoint_fields={"training_data": "iam"},
    )

    assert len(history["validation_cer"]) == 2

    with open(history_path) as file:
        assert len(list(csv.reader(file))) == 3  # header + 2 epochs

    # The safe loader must accept it, and the config must be the one
    # the app/evaluation will rebuild.
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert raw["training_data"] == "iam"
    assert raw["image_width"] == 256 and raw["ctc_timesteps"] == 64

    loaded_model, loaded_vocab, config, _ = hc.load_model(checkpoint_path)
    assert config["image_width"] == 256
    assert loaded_vocab.num_classes == vocabulary.num_classes

    out = loaded_model(torch.rand(1, 1, 128, 256))
    assert out.shape == (64, 1, vocabulary.num_classes)


def test_loader_workers_get_distinct_augmentation_streams(tmp_path):
    vocabulary = iam.build_vocabulary(["abc"])
    dataset = iam.IamLineDataset.from_pairs(
        fake_pairs(["abc"] * 4), vocabulary, image_width=128,
        ctc_timesteps=32, augment=True, rng=random.Random(0),
    )

    class Info:
        pass

    info = Info()
    info.dataset = dataset

    original = torch.utils.data.get_worker_info
    torch.utils.data.get_worker_info = lambda: info
    try:
        train_iam._seed_worker(0)
        first = dataset.rng.random()
        train_iam._seed_worker(1)
        second = dataset.rng.random()
    finally:
        torch.utils.data.get_worker_info = original

    assert first != second
