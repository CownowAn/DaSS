import os
from tqdm import tqdm
import requests
import zipfile
import argparse


def download_file(url, filename):
  """
  Helper method handling downloading large files from `url`
  to `filename`. Returns a pointer to `filename`.
  """
  chunkSize = 1024
  r = requests.get(url, stream=True)
  with open(filename, 'wb') as f:
    pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
    for chunk in r.iter_content(chunk_size=chunkSize):
      if chunk: # filter out keep-alive new chunks
        pbar.update (len(chunk))
        f.write(chunk)
  return filename

def get_data(dir_path, file_name, url):
    print(f"Downloading {file_name} datasets\n")
    full_name = os.path.join(dir_path, file_name)
    download_file(url, full_name)
    print("Downloading done.\n")
  

def download_tiny_imagenet(dir_path):
  dir_path = os.path.join(dir_path, 'tiny_imagenet')
  os.makedirs(f'{dir_path}/10_split', exist_ok=True)
  os.makedirs(f'{dir_path}/labels', exist_ok=True)
  down_dict = {
    'train_images.npy': 'https://www.dropbox.com/s/pzkc6krcnz124d3/train_images.npy?dl=1',
    'valid_images.npy': 'https://www.dropbox.com/s/mvgwhpl9wktkb8p/valid_images.npy?dl=1',
    'labels/train_labels.npy': 'https://www.dropbox.com/s/ki1l67903ekxwdj/train_labels.npy?dl=1',
    'labels/valid_labels.npy': 'https://www.dropbox.com/s/ta4ulgv8qs9ncsa/valid_labels.npy?dl=1',
    '10_split/split_0.npz': 'https://www.dropbox.com/s/sgmfl54txjo2gnz/split_0.npz?dl=1',
    '10_split/split_1.npz': 'https://www.dropbox.com/s/55l4wz7jqef5t2f/split_1.npz?dl=1',
    '10_split/split_2.npz': 'https://www.dropbox.com/s/suc6t63xmvkfdoj/split_2.npz?dl=1',
    '10_split/split_3.npz': 'https://www.dropbox.com/s/urcsphr5zuxjd6j/split_3.npz?dl=1',
    '10_split/split_4.npz': 'https://www.dropbox.com/s/8vrhfpn1kvkj2pq/split_4.npz?dl=1',
    '10_split/split_5.npz': 'https://www.dropbox.com/s/yycaxhn8ivu4qoo/split_5.npz?dl=1',
    '10_split/split_6.npz': 'https://www.dropbox.com/s/2qygdivnhjq5x30/split_6.npz?dl=1',
    '10_split/split_7.npz': 'https://www.dropbox.com/s/xy2xavssp08rflg/split_7.npz?dl=1',
    '10_split/split_8.npz': 'https://www.dropbox.com/s/ga9u6j4n06epsay/split_8.npz?dl=1',
    '10_split/split_9.npz': 'https://www.dropbox.com/s/wssc1y0sk3uc387/split_9.npz?dl=1',

  }
  for (file_name, url) in down_dict.items():
    get_data(dir_path, file_name, url)


def download_cub(dir_path):
  dir_path = os.path.join(dir_path, 'cub')
  os.makedirs(dir_path, exist_ok=True)
  down_dict = {
    'train_images.npy': 'https://www.dropbox.com/s/kxjc6e6io0cvzmt/train_images.npy?dl=1',
    'test_images.npy': 'https://www.dropbox.com/s/vqs17ei1qda8wkm/test_images.npy?dl=1',
    'train_labels.npy': 'https://www.dropbox.com/s/z1833hmg7vq4ryn/train_labels.npy?dl=1',
    'test_labels.npy': 'https://www.dropbox.com/s/7ye8wayt1ghymnm/test_labels.npy?dl=1',
  }
  for (file_name, url) in down_dict.items():
    get_data(dir_path, file_name, url)


def download_dtd(dir_path):
  dir_path = os.path.join(dir_path, 'dtd')
  os.makedirs(dir_path, exist_ok=True)
  down_dict = {
    'train_images.npy': 'https://www.dropbox.com/s/2r23fiydz5lvt58/train_images.npy?dl=1',
    'test_images.npy': 'https://www.dropbox.com/s/resatuo7h3i6xpo/test_images.npy?dl=1',
    'train_labels.npy': 'https://www.dropbox.com/s/j3ri6ygc13k2o3t/train_labels.npy?dl=1',
    'test_labels.npy': 'https://www.dropbox.com/s/ph5ouk8l47h5vr4/test_labels.npy?dl=1',
  }
  for (file_name, url) in down_dict.items():
    get_data(dir_path, file_name, url)


def download_quickdraw(dir_path):
  dir_path = os.path.join(dir_path, 'quickdraw')
  os.makedirs(dir_path, exist_ok=True)
  down_dict = {
    'train_images.npy': 'https://www.dropbox.com/s/3yrcsor9cukfoa6/train_images.npy?dl=1',
    'test_images.npy': 'https://www.dropbox.com/s/ddd0xv4lsge5b7h/test_images.npy?dl=1',
    'train_labels.npy': 'https://www.dropbox.com/s/9vz2bsjhu4fi7ef/train_labels.npy?dl=1',
    'test_labels.npy': 'https://www.dropbox.com/s/5t4dii2tc5008k8/test_labels.npy?dl=1',
  }
  for (file_name, url) in down_dict.items():
    get_data(dir_path, file_name, url)


def download_stanford_cars(dir_path):
  dir_path = os.path.join(dir_path, 'stanford_cars')
  os.makedirs(dir_path, exist_ok=True)
  down_dict = {
    'train_images.npy': 'https://www.dropbox.com/s/w09ay5q6o4wp9cf/train_images.npy?dl=1',
    'test_images.npy': 'https://www.dropbox.com/s/xcx689plnfkmjmu/test_images.npy?dl=1',
    'train_labels.npy': 'https://www.dropbox.com/s/dhe6bu5hfgar9lv/train_labels.npy?dl=1',
    'test_labels.npy': 'https://www.dropbox.com/s/4fel82itunz52nk/test_labels.npy?dl=1',
  }
  for (file_name, url) in down_dict.items():
    get_data(dir_path, file_name, url)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default='stanford_cars')
  parser.add_argument('--dir_path', type=str, default='dataset')
  args = parser.parse_args()

  os.makedirs(args.dir_path, exist_ok=True)

  if args.name == 'all':
    download_tiny_imagenet(args.dir_path)
    download_cub(args.dir_path)
    download_dtd(args.dir_path)
    download_quickdraw(args.dir_path)
    download_stanford_cars(args.dir_path)

  elif args.name == 'tiny_imagenet':
    download_tiny_imagenet(args.dir_path)
  elif args.name == 'cub':
    download_cub(args.dir_path)
  elif args.name == 'dtd':
    download_dtd(args.dir_path)
  elif args.name == 'quickdraw':
    download_quickdraw(args.dir_path)
  elif args.name == 'stanford_cars':
    download_stanford_cars(args.dir_path)


if __name__ == '__main__':
    main()