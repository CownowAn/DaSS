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

def get_checkpoint(dir_path, file_name, url):
    print(f"Downloading {dir_path}/{file_name} checkpoint\n")
    full_name = os.path.join(dir_path, file_name)
    download_file(url, full_name)
    print("Downloading done.\n")
  

def download_teachers(root_dir_path):
  down_dict = {
    'tiny_imagenet-0': 'https://www.dropbox.com/s/8nvgc483wcun10q/model_best.pth.tar?dl=1',
    'tiny_imagenet-1': 'https://www.dropbox.com/s/69x0aidul17ic2f/model_best.pth.tar?dl=1',
    'tiny_imagenet-2': 'https://www.dropbox.com/s/rzrbzyxc2vxcim3/model_best.pth.tar?dl=1',
    'tiny_imagenet-3': 'https://www.dropbox.com/s/27zdt5w2b4fb72g/model_best.pth.tar?dl=1',
    'tiny_imagenet-4': 'https://www.dropbox.com/s/gbshi45vw9ontoe/model_best.pth.tar?dl=1',
    'tiny_imagenet-5': 'https://www.dropbox.com/s/6bd7j05fqdhk8q8/model_best.pth.tar?dl=1',
    'tiny_imagenet-6': 'https://www.dropbox.com/s/uo61cideecq2fhg/model_best.pth.tar?dl=1',
    'tiny_imagenet-7': 'https://www.dropbox.com/s/voc84si8uly8gwa/model_best.pth.tar?dl=1',
    'tiny_imagenet-8': 'https://www.dropbox.com/s/9p5k0wbq3u4412g/model_best.pth.tar?dl=1',
    'tiny_imagenet-9': 'https://www.dropbox.com/s/npluuu74xxsqm22/model_best.pth.tar?dl=1',
    'cub': 'https://www.dropbox.com/s/k7htlkdjhcosliq/model_best.pth.tar?dl=1',
    'dtd': 'https://www.dropbox.com/s/seir8w9shcel1hd/model_best.pth.tar?dl=1',
    'stanford_cars': 'https://www.dropbox.com/s/9cad7a41soc1zfr/model_best.pth.tar?dl=1',
    'quickdraw': 'https://www.dropbox.com/s/1g5i5obahn327bi/model_best.pth.tar?dl=1',

  }
  for (folder_name, url) in down_dict.items():
    dir_path = os.path.join(root_dir_path, folder_name)
    os.makedirs(dir_path, exist_ok=True)
    file_name = 'model_best.pth.tar'
    get_checkpoint(dir_path, file_name, url)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default='all')
  parser.add_argument('--dir_path', type=str, default='checkpoint/teacher')
  args = parser.parse_args()

  os.makedirs(args.dir_path, exist_ok=True)

  download_teachers(args.dir_path)


if __name__ == '__main__':
    main()