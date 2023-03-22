# import the necessary packages
import argparse
import pickle
import time

import cv2

from pyimagesearch.hashing import convert_image

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tree", required=True, type=str,
                help="path to pre-constructed VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str,
                help="path to hashes dictionary")
ap.add_argument("-q", "--query", required=True, type=str,
                help="path to input query image")
ap.add_argument("-d", "--distance", type=int, default=10,
                help="maximum hamming distance")
args = vars(ap.parse_args())

# load the VP-Tree and hashes dictionary
print("[INFO] loading VP-Tree...")
tree = pickle.loads(open(args["tree"], "rb").read())
# load the input query image
image = cv2.imread(args["query"])
cv2.imshow("Query", image)
# compute the hash for the query image, then convert it
query = {'image': convert_image(image)}

# perform the search
print("[INFO] performing search...")
start = time.time()
distance, nearest = tree.get_nearest_neighbor(query)
end = time.time()
print("[INFO] search took {} seconds".format(end - start))
resultPath = nearest['path']
# load the result image and display it to our screen
result = cv2.imread(resultPath)
cv2.imshow("Result", result)
cv2.waitKey(0)
