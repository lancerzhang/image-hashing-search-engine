# import the necessary packages
from pyimagesearch.hashing import convert_hash
from pyimagesearch.hashing import hamming
from pyimagesearch.hashing import dhash
from imutils import paths
import argparse
import pickle
import vptree
import cv2
import tensorflow_datasets as tfds

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, type=str,
                help="tensorflow dataset name")
ap.add_argument("-t", "--tree", required=True, type=str,
                help="path to output VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str,
                help="path to output hashes dictionary")
args = vars(ap.parse_args())

hashes = {}
dataset_name = args["dataset"]
ds = tfds.load(name=dataset_name, split="train")
for item in ds:
    image = item['image'].numpy()
    h = dhash(image)
    h = convert_hash(h)
    # update the hashes dictionary
    l = hashes.get(h, [])
    l.append(image)
    hashes[h] = l

# build the VP-Tree
print("[INFO] building VP-Tree...")
points = list(hashes.keys())
tree = vptree.VPTree(points, hamming)

# serialize the VP-Tree to disk
print("[INFO] serializing VP-Tree...")
f = open(args["tree"], "wb")
f.write(pickle.dumps(tree))
f.close()
# serialize the hashes to dictionary
print("[INFO] serializing hashes...")
f = open(args["hashes"], "wb")
f.write(pickle.dumps(hashes))
f.close()
