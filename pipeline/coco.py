import os
import urllib.request
from tqdm import tqdm
import tarfile
import zipfile
import json

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def _maybe_download_and_extract(download_dir, url):
    filename = url.split('/')[-1]
    file_path = os.path.join(download_dir, filename)

    if not os.path.exists(file_path):
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
              desc=url.split('/')[-1]) as t:
            file_path, _ = urllib.request.urlretrieve(url=url,
                                                      filename=file_path,
                                                      reporthook=t.update_to)
        print()
        print("Download finished. Extracting files.")

        if file_path.endswith(".zip"):
            zipfile.ZipFile(file=file_path, mode="r").extractall(download_dir)
        elif file_path.endswith((".tar.gz", ".tgz")):
            tarfile.open(name=file_path, mode="r:gz").extractall(download_dir)

        print("Done.")
    else:
        print("Data has apparently already been downloaded.")

def maybe_download_and_extract(path):
    filenames = ["zips/train2017.zip",
                 "zips/val2017.zip",
                 "annotations/annotations_trainval2017.zip"]
    data_url = "http://images.cocodataset.org/"

    for filename in filenames:
        url = data_url + filename
        _maybe_download_and_extract(path, url)

def load_records(data_dir, train=True):
    if train:
        filename = "captions_train2017.json"
        mid_path = 'train2017'
    else:
        filename = "captions_val2017.json"
        mid_path = 'val2017'

    path = os.path.join(data_dir, "annotations", filename)

    with open(path, "r", encoding="utf-8") as file:
        data_raw = json.load(file)

    images = data_raw['images']
    annotations = data_raw['annotations']

    records = dict()

    for image in images:
        image_id = image['id']
        filename = os.path.join(data_dir, mid_path, image['file_name'])

        record = dict()

        record['filename'] = filename

        record['captions'] = list()

        records[image_id] = record

    for ann in annotations:
        image_id = ann['image_id']
        caption = ann['caption']

        record = records[image_id]
        record['captions'].append(caption)

    records_list = [(key, record['filename'], record['captions'])
                    for key, record in sorted(records.items())]

    ids, filenames, captions = zip(*records_list)

    return ids, filenames, captions
