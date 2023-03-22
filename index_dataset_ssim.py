# import the necessary packages
import argparse
import pickle

import tensorflow_datasets as tfds
import vptree

from pyimagesearch.hashing import get_hash, ssim_compare, convert_image

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
    origin = item['image'].numpy()
    image = convert_image(origin)
    # compute the hash for the image and convert it
    h = get_hash(image)
    hashes[h] = {'hash': h, 'image': image, 'origin': origin}

# build the VP-Tree
print("[INFO] building VP-Tree...")
points = list(hashes.values())
tree = vptree.VPTree(points, ssim_compare)

# serialize the VP-Tree to disk
print("[INFO] serializing VP-Tree...")
f = open(args["tree"], "wb")
f.write(pickle.dumps(tree))
f.close()
