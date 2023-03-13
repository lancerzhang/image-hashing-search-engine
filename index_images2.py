# import the necessary packages
from pyimagesearch.hashing import convert_hash, get_hash, ssim_compare, convert_image
from pyimagesearch.hashing import hamming
from pyimagesearch.hashing import dhash
from imutils import paths
import argparse
import pickle
import vptree
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, type=str,
                help="path to input directory of images")
ap.add_argument("-t", "--tree", required=True, type=str,
                help="path to output VP-Tree")
ap.add_argument("-a", "--hashes", required=True, type=str,
                help="path to output hashes dictionary")
args = vars(ap.parse_args())

# grab the paths to the input images and initialize the dictionary
# of hashes
imagePaths = list(paths.list_images(args["images"]))
hashes = {}
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    image = cv2.imread(imagePath)
    image = convert_image(image)
    # compute the hash for the image and convert it
    h = get_hash(image)
    hashes[h] = {'hash': h, 'path': imagePath, 'image': image}

# build the VP-Tree
print("[INFO] building VP-Tree...")
points = list(hashes.values())
tree = vptree.VPTree(points, ssim_compare)

# serialize the VP-Tree to disk
print("[INFO] serializing VP-Tree...")
f = open(args["tree"], "wb")
f.write(pickle.dumps(tree))
f.close()
# serialize the hashes to dictionary
# print("[INFO] serializing hashes...")
# f = open(args["hashes"], "wb")
# f.write(pickle.dumps(hashes))
# f.close()
