# image-hashing-search-engine

## Guide
https://pyimagesearch.com/2019/08/26/building-an-image-hashing-search-engine-with-vp-trees-and-opencv/

Index images
```shell
time python index_images.py --images 101_ObjectCategories --tree vptree.pickle --hashes hashes.pickle
```

Search image
```shell
python search.py --tree vptree.pickle --hashes hashes.pickle --query queries/buddha.jpg
```

## Data set
https://data.caltech.edu/records/mzrjq-6wc02

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