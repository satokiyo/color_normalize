import os
import glob
import re
import pandas as pd

from tqdm import tqdm
from datetime import datetime, timedelta

# ファイル名からクラスを取得するための正規表現
LABEL_PAT = re.compile(r'\w+_class_(?P<label>\d)')
def extract_label(path):
    """ファイル名からクラスラベルを取得する
    path: ファイルのパス
    例）
    >>> path = 'slide00_x_y_class_1.jpg'
    >>> extract_label(path)
    1
    """
    basename = os.path.basename(path)
    result = LABEL_PAT.search(basename)
    if result is not None:
        return int(result.group('label'))

    else:
        return -1


def divide_sample_by_label(samples, labels):
    """
    クラスラベルごとに画像ファイルパスをわけて、まとめたdictを返す
    キー　　: クラスラベル
    バリュー: 画像ファイルパス
    """
    sample_dict = {}
    for sample, label in zip(samples, labels):
        sample_dict.setdefault(label, []).append(sample)

    return sample_dict


def define_directory(parent, *children):
    directory = os.path.join(parent, *children) if children else parent
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory

IMAGE_EXTENSIONS = ('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.gif', '.bmp',
                    '.TIF', '.TIFF', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP')
def get_imagefiles(dirpath):
    imagefile_list = list()
    for fname in tqdm(glob.glob('{}/**'.format(dirpath), recursive=True), ascii=True):
        if os.path.isfile(fname):
            _, ext = os.path.splitext(fname)
            if ext in IMAGE_EXTENSIONS:
                imagefile_list.append(os.path.abspath(fname))
  
    return imagefile_list

def calc_deltatime(start, end):
    s_td = timedelta(seconds=start)
    e_td = timedelta(seconds=end)
    elapsed_td = e_td - s_td
    seconds = elapsed_td.seconds
    hours = seconds // 3600
    minutes = seconds // 60 - hours * 60
    seconds -= (hours * 3600 + minutes * 60)
    return elapsed_td.days, hours, minutes, seconds


def write_csv(iterable, filepath, index=False):
    df_for_csv = pd.DataFrame(iterable)
    df_for_csv.to_csv(filepath, index=index, encoding='utf-8')