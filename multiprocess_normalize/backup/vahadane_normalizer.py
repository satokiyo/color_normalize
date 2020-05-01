import itertools
import time
from concurrent import futures
from inspect import isgenerator
from logging import getLogger, DEBUG
from functools import partial

import numpy as np
import spams
import staintools
from staintools import LuminosityStandardizer
from staintools.tissue_masks.luminosity_threshold_tissue_locator import LuminosityThresholdTissueLocator
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.miscellaneous.optical_density_conversion import convert_OD_to_RGB
from staintools.miscellaneous.optical_density_conversion import convert_RGB_to_OD
from staintools.miscellaneous.get_concentrations import get_concentrations
from staintools.miscellaneous.exceptions import TissueMaskException
from tqdm import tqdm

#from .logger_setting import set_logger
from logger_setting import set_logger


class ConcurrentVahadaneStainNormalizer:
    """スレッドで処理を行う色正規化器
    次の論文の手法を参考に実装
    A. Vahadane et al. 'Structure-Preserving Color Normalization and 
                        Sparse Stain Separation for Histological Images'

    計算するパラメータはW, H, RM_H
        W: 色基底行列
        H: 色強度行列
        RM_H or RM(H): Hの99%の値をとったもの

    Usage:
        >>> normalizer = ConcurrentVahadaneStainNormalizer()
        >>> normalizer.fit(target_images)
        >>> normalized_I = normalizer.transform(I)
    """

    def __init__(self, target_W_file=None, target_RM_H_file=None, pool=None):
        """すでに計算済みのパラメータを読み込む
        targetとは「正規化する際に目標とする色合いを持つ画像」のこと

        Args:
            target_W_file: 色基底行列パラメータ.npyファイルのパス
            target_RM_H_file: 色強度行列パラメータ.npyファイルのパス
            workers: 並列化するスレッドの数
        """
        self._logger = getLogger('{}.{}'.format(__name__, 
                                                self.__class__.__name__))

        if pool is None:
            self.pool = futures.ProcessPoolExecutor()

        else:
            self.pool = pool

        self.target_W = None
        if target_W_file:
            msg = 'Load "{}" to target_W'
            self._logger.debug(msg.format(target_W_file))
            self.target_W = np.load(target_W_file)

        self.target_RM_H = None
        if target_RM_H_file:
            msg = 'Load "{}" to target_RM_H'
            self._logger.debug(msg.format(target_RM_H_file))
            self.target_RM_H = np.load(target_RM_H_file)


    @property
    def is_fit(self):
        """fitメソッドを実行する必要があるのかのフラグ

        Returns:
            必要がないのであればTrue, それ以外はFalseを返す
        """
        return self.target_W is not None and self.target_RM_H is not None

    
    def fit(self, target_images):
        """targetに合わせた色正規化を行うために、パラメータを計算する

        Args:
            target_images: 色正規化の対象となる画像
                （numpy.ndarrayかnumpy.ndarrayのジェネレータ）
        """
        if isgenerator(target_images):
            # すべての画像から基底行列Wと強度行列Hを求める
            t_W_gen, t_H_gen = self._generate_W_and_H(target_images)

            # すべての画像から書く染料の基底ベクトルを計算
            hematoxylin_base_vec_list = []
            eosin_base_vec_list = []
            t0 = time.time()
            for t_W in t_W_gen:
                hematoxylin_base_vec = t_W[0, :]
                eosin_base_vec = t_W[1, :]
                hematoxylin_base_vec_list.append(hematoxylin_base_vec)
                eosin_base_vec_list.append(eosin_base_vec)

            ROIs_elapsed = time.time() - t0
            time_msg = '[{:.8f}s] {} elapsed'
            self._logger.debug(time_msg.format(ROIs_elapsed, 'ROIs'))

            # ROIごとの染料の基底ベクトル集合の各要素の中央値を求める
            # それを代表の基底行列Wとする
            t0 = time.time()
            hematoxylin_base_vecs = np.array(hematoxylin_base_vec_list)
            eosin_base_vecs = np.array(eosin_base_vec_list)
            median_hematoxylin = np.median(hematoxylin_base_vecs, axis=0)
            median_eosin = np.median(eosin_base_vecs, axis=0)
            self.target_W = np.vstack([normalize(median_hematoxylin),
                                       normalize(median_eosin)])
            W_elapsed = time.time() - t0
            self._logger.debug(time_msg.format(W_elapsed, 'target W'))

            # RM(H)を求める
            t0 = time.time()
            self.target_RM_H = self._robust_pseudo_maximaum(t_H_gen)
            H_elapsed = time.time() - t0
            self._logger.debug(time_msg.format(H_elapsed, 'target RM(H)'))

            print('target W\n{}'.format(self.target_W))
            print('target RM(H)\n{}'.format(self.target_RM_H))

        else:
            # 単一のROIからパラメータを求める
            target_W = calculate_normalized_W(target_images)
            hematoxylin_base_vec = normalize(target_W[0, :])
            eosin_base_vec = normalize(target_W[1, :])
            self.target_W = np.vstack([hematoxylin_base_vec, eosin_base_vec])
            target_H = get_concentrations(target_images, self.target_W)
            self.target_RM_H = robust_pseudo_maximaum(target_H)


    def transform(self, I, norm_source_H):
        """色正規化を行う

        Args:
            I: 正規化したい画像(numpy.ndarray)
            norm_source_H: Source画像の正規化された色強度パラメータ
        Returns:
            色正規化後の画像
        """
        try:
            norm_I = 255 * np.exp(-1 * np.dot(norm_source_H, self.target_W))

        except Exception:
            return I

        else:
            return norm_I.reshape(I.shape).astype(np.uint8)



    def _generate_W_and_H(self, images):
        """imagesから各々のパラメータW, Hを取得する

        Args:
            images: パラメータを取得する画像群
                （(画像パス, numpy.ndarray)のジェネレータ）
        Returns:
            Wのジェネレータと、Hのジェネレータ
        """
        W_and_I_generator = self.generate_W_and_I(images)
        W_and_I_gen1, W_and_I_gen2 = itertools.tee(W_and_I_generator)
        W_generator = (W for W, _ in W_and_I_gen1)
        H_generator = self.generate_H(W_and_I_gen2)
        return W_generator, H_generator

    def generate_W_and_I(self, images):
        """パラメータWと画像のタプルのジェネレータを作成

        Args:
            images: 画像のジェネレータ
        Returns:
            (W, 画像)のジェネレータ
        """
        imagepaths = image_generator_to_imagepaths(images)
        mappings = zip(imagepaths,
                       self.pool.map(calculate_W, imagepaths))
        for imagepath, W in mappings:
            yield W, read_image(imagepath)


    def generate_H(self, W_and_I_generator):
        """パラメータHのジェネレータを作成

        Args:
            W_and_I_generator: (W, 画像)のジェネレータ
        Returns:
            Hのジェネレータ
        """
        H_generator = (get_concentrations(I, W) for W, I in W_and_I_generator)
        return H_generator



    def _robust_pseudo_maximaum(self, H_gen):
        """ROIごとの強度行列の集合からRM(H)を計算
        99%点の値を代表の強度行列RM(H)とする

        Args:
            H_gen: 強度行列Hのジェネレータ
        Returns:
            RM(H)
        """
        self._logger.info('calculate RM(H)')
        hematoxylin_density_vec_list = []
        eosin_density_vec_list = []
        for H in H_gen:
            hematoxylin_density_vec = H[:, 0]
            eosin_density_vec = H[:, 1]
            hematoxylin_density_vec_list.append(hematoxylin_density_vec)
            eosin_density_vec_list.append(eosin_density_vec)

        hematoxylin_density_vecs = np.array(hematoxylin_density_vec_list)
        eosin_density_vecs = np.array(eosin_density_vec_list)
        RM_hematoxylin_density = np.percentile(hematoxylin_density_vecs, 99)
        RM_eosin_density = np.percentile(eosin_density_vecs, 99)
        return np.array([RM_hematoxylin_density,
                         RM_eosin_density]).reshape((1, 2))


def image_generator_to_imagepaths(image_generator):
    """(画像パス, 画像)のジェネレータを画像パスのリストに変換する

    Args:
        image_generator: （画像パス, 画像）のジェネレータ
    Returns:
        画像パスのリスト
    """
    imagepaths = [imagepath for imagepath, _ in image_generator]
    return imagepaths


def read_image(imagepath):
    """画像の読み込みを行う

    Args:
        imagepath: 画像パス
    Returns:
        画像
    """
    image = staintools.read_image(imagepath)
    image = LuminosityStandardizer.standardize(image)
    return image



def robust_pseudo_maximaum(H):
    """強度行列HからRM(H)を計算

    Args:
        H: 強度行列
    Returns:
        RM(H)
    """
    hematoxylin_density_vec = H[:, 0]
    eosin_density_vec = H[:, 1]
    RM_hematoxylin_density = np.percentile(hematoxylin_density_vec, 99)
    RM_eosin_density = np.percentile(eosin_density_vec, 99)
    return np.array([RM_hematoxylin_density,
                     RM_eosin_density]).reshape((1, 2))


def calculate_normalized_W(image):
    """正規化されたWを計算

    Args:
        image: 画像(numpy.ndarray)
    Returns:
        正規化されたパラメータW
    """
    return VahadaneStainExtractor.get_stain_matrix(image)


def calculate_W(imagepath, luminosity_threshold=0.8, regularizer=0.1):
    """Wを計算する

    Args:
        imagepath: 画像パス
        luminosity_threshold:
        regularizer:

    Returns:
        パラメータW
    """
    I = read_image(imagepath)
    tissue_mask = LuminosityThresholdTissueLocator.get_tissue_mask(
        I, luminosity_threshold=luminosity_threshold).reshape((-1, ))

    od = convert_RGB_to_OD(I).reshape((-1, 3))
    od = od[tissue_mask]

    dictionary = spams.trainDL(X=od.T, K=2, lambda1=regularizer,
                               mode=2, modeD=0,
                               posAlpha=True, posD=True, verbose=False).T

    if dictionary[0, 0] < dictionary[1, 0]:
        dictionary = dictionary[[1, 0], :]

    return dictionary

            

def normalize(vec):
    """ベクトルの正規化を行う関数

    Args:
        vec: ベクトル(numpy.ndarray)
    Returns:
        vecを正規化したベクトル
    """
    l2_norm = np.linalg.norm(vec)
    return vec / l2_norm
