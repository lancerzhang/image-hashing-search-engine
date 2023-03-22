# import the necessary packages
import random

from pyimagesearch.hashing import convert_hash
from pyimagesearch.hashing import dhash
import argparse
import pickle
import time
import cv2
import tensorflow_datasets as tfds

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tree", required=True, type=str,
                help="path to pre-constructed VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str,
                help="path to hashes dictionary")
ap.add_argument("-q", "--dataset", required=True, type=str,
                help="tensorflow dataset name")
ap.add_argument("-d", "--distance", type=int, default=10,
                help="maximum hamming distance")
args = vars(ap.parse_args())

# load the VP-Tree and hashes dictionary
print("[INFO] loading VP-Tree and hashes...")
tree = pickle.loads(open(args["tree"], "rb").read())
hashes = pickle.loads(open(args["hashes"], "rb").read())
# load the input query image

dataset_name = args["dataset"]
ds = tfds.load(name=dataset_name, split="train")
ds_numpy = tfds.as_numpy(ds)
query_index = random.randint(0, len(ds_numpy))
image = list(ds_numpy)[query_index ]['image']
# compute the hash for the query image, then convert it
queryHash = dhash(image)
queryHash = convert_hash(queryHash)

# perform the search
print("[INFO] performing search...")
start = time.time()
results = tree.get_all_in_range(queryHash, args["distance"])
end = time.time()
print("[INFO] search took {} seconds".format(end - start))
results = sorted(results)
# only should top 5 results
results = results[:5]

# loop over the results
for (d, h) in results:
    # grab all image paths in our dataset with the same hash
    resultImages = hashes.get(h, [])
    # loop over the result paths
    for img in resultImages:
        cv2.imshow("Result", img)
        cv2.waitKey(0)
