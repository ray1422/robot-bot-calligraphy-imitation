# (c) 2021 Tian-Rui Chung
# this is single-use code to preprocess the data for the simulation
# don't waste time to reusing it

from base64 import urlsafe_b64encode
import errno
import os
import glob
import shutil
import random

DATASET_RAW_DIR = "./datasets/sim_raw"
DATASET_TRACE_DIR = "./datasets/trace_6d"
DATASET_PROCESSED = "./datasets/sim"


for f in glob.glob(os.path.join(DATASET_RAW_DIR, "*")):
    if not os.path.isdir(f):
        continue

    data_char = os.path.basename(f)
    # urlsafe base64 encode
    char_id = urlsafe_b64encode(data_char.encode("utf-8")).decode("utf-8")
    # /<data_char>/char*_stroke/ is the stroke dir. We need to extract the name and \
    # find the corresponding trace in DATA_TRACE_DIR.
    try:
        stroke_name = glob.glob(os.path.join(f, "char*_stroke"))[0]
        stroke_name = os.path.basename(stroke_name)
    except IndexError:
        print("corrupted data: not contain char stroke id", data_char)
        continue

    # find the corresponding trace
    stroke_trace_filename = os.path.join(DATASET_TRACE_DIR, f"{stroke_name}.txt")
    if not os.path.exists(stroke_trace_filename):
        print("corrupted data: can't find the trace file for", data_char, stroke_trace_filename)
        continue

    try:
        os.makedirs(os.path.join(DATASET_PROCESSED, char_id))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # copy the trace file to the processed dir
    shutil.copy(stroke_trace_filename, os.path.join(DATASET_PROCESSED, char_id, "trace.txt"))

    # copy the stroke images to the processed dir
    # the filenames are "tmp1_{i}.png"
    for i in range(0, 100):
        stroke_img_filename = os.path.join(DATASET_RAW_DIR, f"{data_char}", f"tmp1_{i}.jpg")
        if not os.path.exists(stroke_img_filename):
            if len(glob.glob(f"{DATASET_RAW_DIR}/{data_char}/*.jpg")) != i or i == 0:
                print("corrupted data: not enough stroke images. probably mis-indexed the strokes!",
                      data_char, stroke_img_filename)
            break
        shutil.copy(stroke_img_filename, os.path.join(DATASET_PROCESSED, char_id, f"stroke_{i}.jpg"))

    # copy the full image to the processed dir
    full_img_filename = os.path.join(DATASET_RAW_DIR, data_char, f"{stroke_name}", "full.png")
    if not os.path.exists(full_img_filename):
        print("corrupted data: not contain full image", data_char, full_img_filename)
        continue
    shutil.copy(full_img_filename, os.path.join(DATASET_PROCESSED, char_id, "full.png"))

    # done!

# split the dataset into train/test set


chars_id = [f for f in os.listdir(DATASET_PROCESSED) if os.path.isdir(os.path.join(DATASET_PROCESSED, f))]
random.shuffle(chars_id)
split_idx = int(len(chars_id) * 0.8)
train_set = chars_id[:split_idx]
test_set = chars_id[split_idx:]
# write the train/test set to file
with open(os.path.join(DATASET_PROCESSED, "train.txt"), "w") as f:
    f.write("\n".join(train_set))

with open(os.path.join(DATASET_PROCESSED, "test.txt"), "w") as f:
    f.write("\n".join(test_set))

print(f"done, {len(train_set)} train, {len(test_set)} test")