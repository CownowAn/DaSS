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
    print(f"Downloading {dir_path}/{file_name} checkpoint\n")
    full_name = os.path.join(dir_path, file_name)
    download_file(url, full_name)
    print("Downloading done.\n")
  

def download_preprocessed(root_dir_path):
  down_dict = {
    'preprocessed.zip': 'https://www.dropbox.com/s/09kn998qbflyozy/preprocessed.zip?dl=1',

  }
  for (file_name, url) in down_dict.items():
    dir_path = os.path.join(root_dir_path)
    os.makedirs(dir_path, exist_ok=True)
    get_data(dir_path, file_name, url)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str, default='all')
  parser.add_argument('--dir_path', type=str, default='.')
  args = parser.parse_args()

  os.makedirs(args.dir_path, exist_ok=True)

  download_preprocessed(args.dir_path)

  with zipfile.ZipFile('preprocessed.zip', 'r') as zip_ref:
      zip_ref.extractall('.')

if __name__ == '__main__':
    main()