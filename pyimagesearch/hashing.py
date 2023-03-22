# import the necessary packages
import hashlib

from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2


def dhash(image, hashSize=8):
    if image.shape[2] == 1:
        gray = image
    else:
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # resize the grayscale image, adding a single column (width) so we
    # can compute the horizontal gradient
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


def convert_hash(h):
    # convert the hash to NumPy's 64-bit float and then back to
    # Python's built in int
    return int(np.array(h, dtype="float64"))


def hamming(a, b):
    # compute and return the Hamming distance between the integers
    return bin(int(a) ^ int(b)).count("1")


def get_hash(img):
    b = img.view(np.uint8)
    return hashlib.sha1(b).hexdigest()


def ssim_compare(a, b):
    return 1 - ssim(a['image'], b['image'])


def convert_image(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (8, 8))
