"""
First thing to run: checks that the IAM data can be loaded and shows what
it looks like, WITHOUT training anything.

    python src/check_iam.py

It prints the split sizes, the column names the loader detected, a few
transcripts, the character count, how many lines the model's CTC length
would drop, and saves five line images to results/iam_check/ so you can
look at them. Send the printed output to Claude if anything fails or
looks unexpected; the loader's column/split assumptions are UNVERIFIED
until this has run once on your machine.
"""

import argparse
import os

import iam_dataset as iam


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=iam.IAM_DATASET_NAME)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--output-dir", default=os.path.join("results", "iam_check"))
    args = parser.parse_args(argv)

    steps = args.width // 4
    vocabulary = None

    for split in ("train", "validation", "test"):
        try:
            data = iam.load_split(split, args.dataset)
        except iam.IamUnavailableError as error:
            raise SystemExit(f"\nERROR: {error}")

        image_column, text_column = iam.detect_columns(data.column_names)
        texts = [iam.normalize_text(t) for t in data[text_column]]

        print(f"\n[{split}] {len(data)} lines; columns: {data.column_names}")
        print(f"  image column: {image_column!r}; text column: {text_column!r}")

        too_long = sum(iam.required_timesteps(t) > steps for t in texts)
        empty = sum(not t for t in texts)
        print(f"  empty: {empty}; longer than {steps} CTC steps: {too_long}")
        print(f"  longest transcript: {max(map(len, texts))} characters")

        for text in texts[:3]:
            print(f"  e.g. {text!r}")

        if split == "train":
            vocabulary = iam.build_vocabulary(texts)
            print(f"  characters: {vocabulary.num_characters}: "
                  f"{''.join(vocabulary.char_to_id)!r}")
            os.makedirs(args.output_dir, exist_ok=True)

            for i in range(5):
                image = data[i][image_column]
                path = os.path.join(args.output_dir, f"train_{i}.png")
                image.save(path)
                print(f"  saved {path} ({image.size[0]}x{image.size[1]}) -> {texts[i]!r}")
        else:
            unknown = sum(
                any(c not in vocabulary.char_to_id for c in t) for t in texts
            )
            print(f"  lines with a character not in the train vocabulary: {unknown}")

    print("\nIf this all looks right, start training (see IAM_TRAINING.md).")


if __name__ == "__main__":
    main()
