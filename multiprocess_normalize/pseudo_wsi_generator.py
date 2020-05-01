import re
import os
import pathlib

from collections import OrderedDict, namedtuple
from logging import getLogger

#from .image_generator import ImageGenerator
from image_generator import ImageGenerator


class PseudoWSIGenerator:
    
    ROI_INFO_PAT = re.compile(r'(?P<slide>\w+)_(?P<x>\d+)_(?P<y>\d+)_class_(?P<label>\d)')
    ROI_INFO_FIELD = ('slide', 'x', 'y', 'label')
    ROIInfo = namedtuple('ROIInfo', ROI_INFO_FIELD)

    SLIDE_SOURCE_PAT = re.compile(r'([A-Z]+)?\d+')

    UNKNOWN_SLIDE = 'unknown'

    def __init__(self, image_generator, image_source_path):
        self._logger = getLogger('__main__.PseudoWSIGenerator')

        self._image_generator = image_generator

        self._slide_dir_list = [d for d in self._image_generator.get_dir_list(image_source_path) 
                                if self.SLIDE_SOURCE_PAT.search(d)]
        self._slide_to_origin = {self.UNKNOWN_SLIDE: self.UNKNOWN_SLIDE}
        for slide_dir in self._slide_dir_list:
            origin_dir, slide_name = os.path.split(slide_dir)
            origin_name = os.path.basename(origin_dir)
            self._slide_to_origin[slide_name] = origin_name



    def generate_map(self, path):
        """
        スライドごとにROI画像ファイルをまとめて、
        スライド名がキーで、スライド内のすべてのROI画像データのジェネレータが値のdictを作成して返す
        """
        slide_map = OrderedDict()
        if self._image_generator.is_directory(path):
            group_dict = self._grouping_path(path)
            for slide_name, slide_dir in group_dict.items():
                slide_map[slide_name] = self._image_generator.generate_images(slide_dir)

        else:
            slide_name = self.get_slide_name(path)
            slide_dir  = self._get_slide_dir(path)
            slide_map[slide_name] = self._image_generator.generate_images(slide_dir)

        return slide_map


    def get_origin(self, slide_name):
        if slide_name in self._slide_to_origin:
            return self._slide_to_origin[slide_name]

        else:
            return slide_name


    @staticmethod
    def get_slide_name(path):
        """
        ファイルパスからスライド名を取得する
        """
        result = PseudoWSIGenerator.ROI_INFO_PAT.search(path)
        if result is not None:
            return PseudoWSIGenerator.ROIInfo(*result.groups()).slide

        else:
            return pathlib.Path(path).parent.name


    def _get_slide_dir(self, path):
        """
        ファイルパスからそのファイルが所属するスライドのディレクトリを取得する
        """
        slide_name = self.get_slide_name(path)
        for slide_dir in self._slide_dir_list:
            basename = os.path.basename(slide_dir)
            if basename == slide_name:
                return slide_dir




    def _grouping_path(self, dirname):
        image_gen = self._image_generator.generate_images(dirname)
        group_dict = {}
        for image_path, _ in image_gen:
            slide_name = self.get_slide_name(image_path)
            if slide_name not in group_dict:
                group_dict[slide_name] = self._get_slide_dir(image_path)

        return group_dict
