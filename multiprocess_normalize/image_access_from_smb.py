import platform
import io
import os
import re
import cv2
import numpy as np

from smb.SMBConnection import SMBConnection
from urllib.parse import urlparse
from collections import namedtuple
from functools import partial
from pathlib import Path
from tqdm import tqdm
from logging import getLogger

#from .image_processing_util import bytes_to_RGB, encode_by_png
from image_processing_util import bytes_to_RGB, encode_by_png


class ImageAccessorFromSMB:

    SMBPathInfo = namedtuple('SMBPathInfo', ['scheme', 'remote_name', 'service_name', 'subpath'])

    AUTH_FILED = ['username', 'password', 'ip']
    AuthInfo   = namedtuple('AuthInfo', AUTH_FILED)

    SERVICE = 1
    SUBPATH = slice(2, None)

    PORT = 139

    EXCEPTED_DIRS = ['.', '..']

    

    def __init__(self, cfg_file):
        self._logger = getLogger('__main__.ImageAccessorFromSMB')

        auth_info = self._read_cfg_file(cfg_file)

        self._ip = auth_info.ip
        self._make_connection = partial(SMBConnection,
                                        auth_info.username, auth_info.password, platform.uname().node)


    def get_image_generator(self, directory):
        """directory配下のすべてのファイルを読み込み、
        （画像ファイル名, 画像データ: numpy.ndarray）のタプルのジェネレータを返す
        """
        smb_path_info = self._to_smb_path_info(directory)
        scheme, remote_name, service_name, subdir = smb_path_info
        with self._make_connection(remote_name) as conn:
            conn.connect(self._ip, self.PORT)
            image_files = self._get_all_files(conn, service_name, subdir)
            for image_file in image_files:
                img_data = self._read_image(conn, service_name, image_file)
                filename = self._abspath(scheme, remote_name, service_name, image_file)
                yield filename, img_data


    def get_dir_list(self, directory):
        """directory配下のすべてのディレクトリを読み込み、そのディレクトリのパスのリストを返す
        """
        smb_path_info = self._to_smb_path_info(directory)
        scheme, remote_name, service_name, subdir = smb_path_info
        with self._make_connection(remote_name) as conn:
            conn.connect(self._ip, self.PORT)
            directory_generator = self._get_all_directories(conn, service_name, subdir)
            return [self._abspath(scheme, remote_name, service_name, dirname) for dirname in directory_generator]


    def get_imagefile_list(self, directory):
        smb_path_info = self._to_smb_path_info(directory)
        scheme, remote_name, service_name, subdir = smb_path_info
        with self._make_connection(remote_name) as conn:
            conn.connect(self._ip, self.PORT)
            imagefiles = self._get_all_files(conn, service_name, subdir)
            imagefile_list = [self._abspath(scheme, remote_name, service_name, imagefile) for imagefile in imagefiles]
            
        return imagefile_list
            



    def get_image(self, filepath):
        """filepathで与えられた画像データ: numpy.ndarrayを返す
        """
        smb_path_info = self._to_smb_path_info(filepath)
        with self._make_connection(smb_path_info.remote_name) as conn:
            conn.connect(self._ip, self.PORT)
            img_data = self._read_image(conn, smb_path_info.service_name, smb_path_info.subpath)

        return img_data


    def is_directory(self, path):
        """ファイルサーバ内のpathがディレクトリかを判定する"""
        smb_path_info = self._to_smb_path_info(path)
        with self._make_connection(smb_path_info.remote_name) as conn:
            conn.connect(self._ip, self.PORT)
            is_directory = conn.getAttributes(smb_path_info.service_name, smb_path_info.subpath).isDirectory

        return is_directory


    def save_images(self, dir_to_save, images):
        """dir_to_save下にimagesで与えられた
        （保存したい画像ファイルパス名, 画像データ）のジェネレータから得られる画像データを所与の名前で保存する
        """
        smb_path_info = self._to_smb_path_info(dir_to_save)
        scheme, remote_name, service_name, subdir_to_save = smb_path_info
        with self._make_connection(remote_name) as conn:
            conn.connect(self._ip, self.PORT)
            for sub_path, image in tqdm(images, ascii=True):
                # 保存先のディレクトリの確認と、ファイル名の変更のためにパスを分ける
                image_dir, image_name = os.path.split(sub_path)
                save_image_dir = os.path.join(subdir_to_save, image_dir)
                if not self._exists(conn, service_name, save_image_dir):
                    self._makedirs(conn, service_name, save_image_dir)

                # pngでのエンコードなのは、エンコード前後でデータの中身が変わらないため
                data = encode_by_png(image)
                image_name = image_name.replace('jpg', 'png')

                image_subpath = os.path.join(save_image_dir, image_name)
                conn.storeFile(service_name, image_subpath, io.BytesIO(data))
                saved_path = self._abspath(scheme, remote_name, service_name, image_subpath)
                self._logger.debug('save to {}...'.format(saved_path))



    # 以下ヘルパーメソッド ###################################################

    def _read_cfg_file(self, cfg_file):
        split_pat = re.compile(r',\s?')

        with open(cfg_file, mode='r', encoding='utf-8') as fp:
            lines = [split_pat.split(line.strip())[1] for line in fp.readlines()]

        if len(lines) > 3:
            lines = lines[:3]

        return self.AuthInfo(*lines)


    def _read_image(self, conn, service_name, subpath):
        with io.BytesIO() as bio:
            conn.retrieveFile(service_name, subpath, bio)
            bio.seek(0)
            img_data = bytes_to_RGB(bio.read())

        return img_data


    def _get_all_files(self, conn, service_name, target_path):
        shared_file_list = conn.listPath(service_name, target_path)
        for sf in (_sf for _sf in shared_file_list if _sf.filename not in self.EXCEPTED_DIRS):
            path = os.path.join(target_path, sf.filename)
            if sf.isDirectory:
                yield from self._get_all_files(conn, service_name, path)

            else:
                yield path

    def _get_all_directories(self, conn, service_name, target_path):
        shared_file_list = conn.listPath(service_name, target_path)
        for sf in (_sf for _sf in shared_file_list if _sf.filename not in self.EXCEPTED_DIRS):
            if sf.isDirectory:
                directory = os.path.join(target_path, sf.filename)
                yield directory
                yield from self._get_all_directories(conn, service_name, directory)


    def _to_smb_path_info(self, path):
        parsed = urlparse(path)
        service_name, subpath = self._split_service_name(parsed.path)
        return self.SMBPathInfo(parsed.scheme, parsed.netloc, service_name, subpath)


    @staticmethod
    def _split_service_name(path):
        splitted_path = path.split('/')
        service_name = splitted_path[ImageAccessorFromSMB.SERVICE]
        subpath = '/'.join(splitted_path[ImageAccessorFromSMB.SUBPATH])
        return service_name, subpath


    def _exists(self, conn, service_name, path):
        p = Path(path)
        parent = p.parent.as_posix().replace('.', '/')
        if parent == '/' or self._exists(conn, service_name, parent):
            return bool([f for f in conn.listPath(service_name, parent) if f.filename == p.name])


    def _makedirs(self, conn, service_name, path):
        parent = Path(path).parent.as_posix().replace('.', '/')
        if not parent == '/' and not self._exists(conn, service_name, parent):
            self._makedirs(conn, service_name, parent)

        if not self._exists(conn, service_name, path):
            conn.createDirectory(service_name, path)


    @staticmethod
    def _abspath(scheme, remote_name, service_name, path):
        return scheme + '://' + os.path.join(remote_name, service_name, path)
