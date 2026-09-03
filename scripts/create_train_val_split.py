#!/usr/bin/env python3
import os
import random
from collections import defaultdict

ROOT = os.path.join('datasets','soybean_aging_R3')
ANNO = os.path.join(ROOT,'anno')
TRAIN_IN = os.path.join(ANNO,'train.txt')
TRAIN_OUT = os.path.join(ANNO,'train_split.txt')
VAL_OUT = os.path.join(ANNO,'val_from_train.txt')
SEED = 42
RATIO = 0.2  # validation fraction


def read_anno(path):
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    items = [l.split() for l in lines]
    return items


def write_lines(path, items):
    with open(path,'w') as f:
        for name,label in items:
            f.write(f"{name} {label}\n")


def main():
    items = read_anno(TRAIN_IN)
    # items are (filename, label)
    by_label = defaultdict(list)
    for name,label in items:
        by_label[label].append((name,label))

    random.seed(SEED)
    train_out = []
    val_out = []

    for label, samples in sorted(by_label.items(), key=lambda x: int(x[0])):
        # shuffle deterministic
        samples_sorted = list(samples)
        random.shuffle(samples_sorted)
        n = len(samples_sorted)
        n_val = max(1, int(round(n * RATIO)))
        # ensure at least 1 in val when possible
        val_samples = samples_sorted[:n_val]
        train_samples = samples_sorted[n_val:]
        # append
        val_out.extend(val_samples)
        train_out.extend(train_samples)

    # sanity counts
    print('total in:', len(items))
    print('train out:', len(train_out))
    print('val out:', len(val_out))

    # write
    write_lines(TRAIN_OUT, train_out)
    write_lines(VAL_OUT, val_out)
    print('Wrote', TRAIN_OUT, VAL_OUT)


if __name__ == '__main__':
    main()
