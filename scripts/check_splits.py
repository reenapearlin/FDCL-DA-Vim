#!/usr/bin/env python3
import os

root = os.path.join('datasets','soybean_aging_R3','anno')
train_f = os.path.join(root,'train.txt')
val_f = os.path.join(root,'val.txt')
test_f = os.path.join(root,'test.txt')

def read_list(p):
    with open(p) as f:
        lines = [l.strip().split()[0] for l in f if l.strip()]
    return lines

train = read_list(train_f)
val = read_list(val_f)
test = read_list(test_f)

print('TRAIN SAMPLES:', len(train))
print('VAL SAMPLES:', len(val))
print('TEST SAMPLES:', len(test))

set_train = set(train)
set_val = set(val)
set_test = set(test)

overlap_train_val = set_train & set_val
overlap_train_test = set_train & set_test
overlap_val_test = set_val & set_test

print('OVERLAP train/val:', len(overlap_train_val))
print('OVERLAP train/test:', len(overlap_train_test))
print('OVERLAP val/test:', len(overlap_val_test))

if overlap_train_val or overlap_train_test or overlap_val_test:
    print('Some overlaps detected. Sample overlaps (up to 10):')
    print('train/val:', list(overlap_train_val)[:10])
    print('train/test:', list(overlap_train_test)[:10])
    print('val/test:', list(overlap_val_test)[:10])
else:
    print('No overlaps detected between splits.')
