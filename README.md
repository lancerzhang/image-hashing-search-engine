# image-hashing-search-engine

## Guide

https://pyimagesearch.com/2019/08/26/building-an-image-hashing-search-engine-with-vp-trees-and-opencv/

Index images

```shell
time python index_images.py --images 101_ObjectCategories --tree vptree.pickle --hashes hashes.pickle
time python index_images_ssim.py --images 101_ObjectCategories --tree vptree.pickle --hashes hashes.pickle
time python index_dataset.py --dataset mnist --tree vptree.pickle --hashes hashes.pickle
time python index_dataset_fn.py --dataset mnist --tree vptree.pickle --dist_fn mse
```

Search image
distance for mse - 2000, ssim - 0.2
```shell
python search.py --tree vptree.pickle --hashes hashes.pickle --query queries/buddha.jpg
python search_ssim.py --tree vptree.pickle --hashes hashes.pickle --query queries/buddha.jpg
python search_dataset.py --tree vptree.pickle --hashes hashes.pickle --dataset mnist
python search_dataset_fn.py --tree vptree.pickle --dataset mnist --distance 2000
```

## Data set

* 101_ObjectCategories, hash is good

https://data.caltech.edu/records/mzrjq-6wc02

* mnist, 8x8 image. dhash & mse is not accurate. ssim is much better but slow, around 100 ms, min 30 ms, max 300 ms to perform
  get_nearest_neighbor()
* mnist, 16x16. mse is good, around 30 ms, min 11 ms, max 59 ms to perform get_nearest_neighbor()

https://www.tensorflow.org/datasets/catalog/mnist

## Performance

https://github.com/idealo/imagededup

### Apple M1 issue

https://github.com/idealo/imagededup/issues/148

Need to manual install.

* Install gcc

```shell
brew install gcc
brew install --cask gcc-aarch64-embedded
```

* Configure gcc https://trinhminhchien.com/install-gcc-g-on-macos-monterey-apple-m1/
* Activate venv

```shell
source ./venv/bin/activate
```

* Switch to another folder and clone imagededup

```shell
git clone https://github.com/idealo/imagededup.git
cd imagededup
pip install "cython>=0.29"
```

* Change `setup.py`

```python
COMPILE_LINK_ARGS = ['-O3', '-mcpu=apple-m1', '-mtune=native']
```

* Install

```shell
python setup.py install
```