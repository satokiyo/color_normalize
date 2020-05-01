import staintools
import glob
import os
import numpy as np

from urllib.parse import urlparse
from functools import partial
from PIL import Image
from logging import getLogger

#from .image_access_from_smb import ImageAccessorFromSMB
from image_access_from_smb import ImageAccessorFromSMB


logger = getLogger(__name__)


def return_only(x):
    return x

class ImageGenerator:

    def __init__(self, cfg_file=None):
        self._logger = getLogger('__main__.ImageGenerator')

        self._image_getter = ImageAccessorFromSMB(cfg_file) if cfg_file else None
        # 関数名が長いので、エイリアスの作成
        self.standardize_luminosity = partial(staintools.LuminosityStandardizer.standardize)


    def generate_images(self, path, std_lum=True):
        """
        与えられたパスのディレクトリ下から、
        すべての画像ファイルのパスと画像データ(numpy.ndarray)のタプルのジェネレータを生成して返す
        """
        transform = self.standardize_luminosity if std_lum else return_only
        if self.is_network_path(path):
            self._logger.debug('{} is network dir...'.format(path))
            for image_path, image in self._image_getter.get_image_generator(path):
                image = transform(image)
                yield image_path, image

        elif os.path.isdir(path):
            self._logger.debug('{} is local dir...'.format(path))
            image_paths = [img for img in glob.glob('{}/**'.format(path), recursive=True) if os.path.isfile(img)]
            for image_path in image_paths:
                image = staintools.read_image(image_path)
                image = transform(image)
                yield image_path, image

        else:
            self._logger.debug('{} is local file'.format(path))
            image = staintools.read_image(path)
            image = transform(image)
            yield path, image


    def get_image(self, path, std_lum=True):
        """
        与えられたパスのファイルから読み込んだ画像データ(numpy.ndarray)を返す
        """
        if self.is_network_path(path):
            image = self._image_getter.get_image(path)
            
        else:
            image = np.array(Image.open(path))
        
        return self.standardize_luminosity(image) if std_lum else image


    def get_dir_list(self, path):
        if self.is_network_path(path):
            return self._image_getter.get_dir_list(path)

        else:
            return [os.path.abspath(d) for d in glob.glob('{}/**'.format(path), recursive=True)
                    if os.path.isdir(d)]


    def get_file_list(self, path):
        if self.is_network_path(path):
            return self._image_getter.get_imagefile_list(path)

        else:
            return [os.path.abspath(f) for f in glob.glob('%s/**' % path, recursive=True)
                                       if os.path.isfile(f)]
    
        
    @staticmethod
    def is_network_path(path):
        return urlparse(path).scheme == 'smb'


    def is_directory(self, path):
        if self.is_network_path(path):
            return self._image_getter.is_directory(path)

        else:
            return os.path.isdir(path)
