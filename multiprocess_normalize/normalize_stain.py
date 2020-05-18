import argparse
import glob
import os
import gc
import sys
import pathlib
import cv2
import re
import tracemalloc
import time
import multiprocessing as mp
import copy
from collections import namedtuple
from pprint import pprint, pformat
from datetime import datetime
from logging import getLogger
from concurrent import futures
from functools import partial

import numpy as np
import pandas as pd
import staintools
from PIL import Image
from tqdm import tqdm
from staintools.miscellaneous.get_concentrations import get_concentrations
from staintools.miscellaneous.exceptions import TissueMaskException

#from .vahadane_normalizer import ConcurrentVahadaneStainNormalizer
#from .vahadane_normalizer import robust_pseudo_maximaum
#from .vahadane_normalizer import calculate_normalized_W
#from .image_generator import ImageGenerator
#from .pseudo_wsi_generator import PseudoWSIGenerator
#from .generic_funcs import get_imagefiles
#from .logger_setting import set_logger

from vahadane_normalizer import ConcurrentVahadaneStainNormalizer
from vahadane_normalizer import robust_pseudo_maximaum
from vahadane_normalizer import calculate_normalized_W

from image_generator import ImageGenerator
from pseudo_wsi_generator import PseudoWSIGenerator
from generic_funcs import get_imagefiles
from logger_setting import set_logger

# 晴山追加
from output_param import output_H
from output_param import output_H_geen_mask
from output_param import make_csv


class ConcurrentStainColorNormalizer:
    """色正規化に必要なTargetのパラメータを計算し提供するクラス

    Grossary:
        Target: 正規化するときに目指す色合いを持つ画像
    """

    def __init__(self, target_path, outdir,
                 target_W=None, target_RM_H=None, pool=None):
        """正規化器の初期化及びTargetのパラメータ計算を行う

        Args:
            target_path: Target画像のパス（ディレクトリかファイルパス）
            outdir: パラメータファイルの保存先ディレクトリ
            target_W: 計算済みTargetのWパラメータのファイルパス
            target_RM_H: 計算済みTargetのRM(H)パラメータのファイルパス
            pool: プロセスプール(ProcessPoolExecutor)
        """        
        self.img_gen = ImageGenerator()

        self.pool = pool
        if self.pool is None:
            self.pool = futures.ProcessPoolExecutor()

        self.normalizer = ConcurrentVahadaneStainNormalizer(
            target_W, target_RM_H, self.pool)

        if not self.normalizer.is_fit:
            target_name = pathlib.Path(target_path).stem
            if pathlib.Path(target_path).is_dir():
                target = self.img_gen.generate_images(target_path)

            else:
                target = self.img_gen.get_image(target_path)

            self.normalizer.fit(target)

            outdir = pathlib.Path(outdir)
            target_matrices_dir = outdir / 'target_matrices' / target_name
            if not target_matrices_dir.exists():
                target_matrices_dir.mkdir(parents=True)

            target_W_path    = target_matrices_dir / 'W.npy'
            target_RM_H_path = target_matrices_dir / 'RM_H.npy'
            np.save(str(target_W_path), self.normalizer.target_W)
            np.save(str(target_RM_H_path), self.normalizer.target_RM_H)

    @property
    def target_W(self):
        return self.normalizer.target_W

    @property
    def target_RM_H(self):
        return self.normalizer.target_RM_H


def read_image(imagepath):
    """画像を読み込む

    Args:
        imagepath: 画像パス
    Returns:
        画像
    """
    image = staintools.read_image(imagepath)
    image = staintools.LuminosityStandardizer.standardize(image)
    return image


#def calculate_source_matrices(source_path, target_RM_H):
#    """Source画像のパラメータの計算のみ行う
#
#    Args:
#        source_image: Source画像
#    Return:
#        正規化された色強度パラメータ
#    """
#    try:
#        source_image = read_image(source_path)
#        source_W = calculate_normalized_W(source_image)
#        source_H = get_concentrations(source_image, source_W)
#        source_RM_H = robust_pseudo_maximaum(source_H)
#        norm_source_H = source_H * (target_RM_H / source_RM_H)
#
#    except TissueMaskException:
#        return None
#
#    else:
#        return norm_source_H


def calculate_source_matrices(req_queue, out_queue, out_queue_flag1, out_queue_flag2, pwsi_gen, save_dir, lk, total_ROIs, msg, msg_fmt, target_RM_H, target_W, column_flag):
    """正規化に必要なSourceのパラメータを計算し、キューに格納する関数

    Args:
        source_ROI: SourceのROI画像パス
        req_queue (multiprocessing.JoinableQueue): 並列処理での共有キュー
    """

###
# 20200518 for single process ver. from
#    while True:
    for _ in range(req_queue.qsize()-1):
# 20200518 for single process ver. to
###
        source_ROI = req_queue.get()
        if source_ROI is None:
            req_queue.task_done()
            break
        try:
            slide_name = pwsi_gen.get_slide_name(source_ROI)
            origin_name = pwsi_gen.get_origin(slide_name)
            source_image = read_image(source_ROI)
            source_W = calculate_normalized_W(source_image)
            source_H = get_concentrations(source_image, source_W)
            source_RM_H = robust_pseudo_maximaum(source_H)
            norm_H = source_H * (target_RM_H / source_RM_H)
            normalize(source_ROI, norm_H, target_W, 
                      slide_name, origin_name, save_dir, lk)

            # 細胞数少ないROIの抽出パラメータ
            vec_thres1_6 = 1.6 # 色強度ベクトルの閾値
            thres_4_0 = 4.0 # 面積比の閾値
            sat_thres_25 = 25 # saturationの閾値
            flag_num = 1 # csv出力の際のフラグ番号
            
            # norm_Hから各画像の細胞数判定結果csvを出力するプログラム
            # vec1.3,1.4,1.5,1.6,1.7 thres4.0 sat25
            output_H(source_ROI, norm_H, slide_name, origin_name, save_dir,
                     out_queue_flag1, vec_thres1_6, thres_4_0, sat_thres_25, flag_num, column_flag)

        except TissueMaskException:
            #この例外が発生したROI画像は正規化されず、正規化後の画像に保存されない。
            #ここでフラグを立てておき、あとでShortTissue_flagと結合して対象外のROIとする。
            #核が少ないフラグを立てる処理はこのROIに対しては行われない
            new_source_ROI = pathlib.Path(source_ROI)
            slide_dir = save_dir / origin_name / slide_name
            image_path = str(slide_dir / new_source_ROI.name)
            flag = 1
            out_queue_flag2.put([image_path, flag])
            continue

        finally:
            # 進捗確認用
            lk.acquire()
            out_queue.put(1)
            if msg:
                sys.stdout.write('\r' + (' ' * len(msg)))
            msg = msg_fmt.format(done=out_queue.qsize(), total=total_ROIs)
            sys.stdout.write('\r' + msg)
            lk.release()

            req_queue.task_done()

    return


def normalize(source_ROI, norm_H, target_W, 
              slide_name, origin_name, save_dir, lk):
    """正規化を行う関数

    Args:
        source_ROI: SourceのROI画像パス
        norm_H (numpy.ndarray): 標準化された色強度行列H
        target_W (numpy.ndarray): Targetの色基底行列W
        slide_name (str): SourceのROIの属するスライド名
        origin_name (str): SourceのROIの属するスライドのグループ名
        save_dir: 正規化後の画像を保存するディレクトリ
    """
    # 正規化
    image = read_image(source_ROI)
    normalized_image = 255 * np.exp(-1 * np.dot(norm_H, target_W))
    normalized_image = normalized_image.reshape(image.shape).astype(np.uint8)

    slide_dir = save_dir / origin_name / slide_name
    lk.acquire()
    if not slide_dir.exists():
        slide_dir.mkdir(parents=True)
    lk.release()

    # 正規化後の画像を保存
    source_ROI = pathlib.Path(source_ROI)
    path_to_save = str(slide_dir / source_ROI.name)
    # 20191115 正規化前の画像を保存しているバグ発見
    # cv2.imwrite(path_to_save, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(path_to_save, cv2.cvtColor(normalized_image, cv2.COLOR_RGB2BGR))


logger = getLogger(__name__)

def main(args):
    outdir = pathlib.Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    logdir = outdir / 'log'
    if not logdir.exists():
        logdir.mkdir(parents=True)
    currdate = datetime.now().strftime('%Y%m%d_%H%M')
    logfile = logdir / currdate
    set_logger(logger, str(logfile))

    target = pathlib.Path(args.target)
    logger.info('[target] {}'.format(target))

    source_list = args.source_list
    logger.info('[source list]\n{}'.format(pformat(source_list)))
    mark = 'Normalize_per_ROI'

    save_dir = pathlib.Path(args.save_dir)
    save_dir = save_dir / mark / target.stem
    logger.info('[save dir] {}'.format(save_dir))

    with futures.ProcessPoolExecutor() as executor:
        normalizer = ConcurrentStainColorNormalizer(
            target, outdir, args.target_W, args.target_RMH, executor)
        
        target_RM_H = normalizer.target_RM_H
        target_W = normalizer.target_W
    
        image_gen = ImageGenerator()
        pwsi_gen = PseudoWSIGenerator(image_gen, args.image_source)
    
        max_workers = mp.cpu_count()
    
        msg_fmt = '{done} / {total} done'
        msg = ''

        #for flag csv
        #column_flag = 1 # csv出力(画像パス、フラグ)
        column_flag = 2 # csv出力(画像パス、フラグ、判定数値(組織上の細胞面積率)、その他)

        # vec1.3,1.4,1.5,1.6,1.7 thres4.0 sat25
        csv_name_vec1_6_thres4_sat25 = "few_cell_flag.csv" # csv名称
        csv_path_vec1_6_thres4_sat25 = make_csv(save_dir, csv_name_vec1_6_thres4_sat25, column_flag)

        name_mask_exp_csv = "tissue_mask_exception_flag.csv" # csv名称
        path_mask_exp_csv = make_csv(save_dir, name_mask_exp_csv, column_flag=1) #column_flag=1としtissue_mask_exception_flag.csvを作成

        if source_list and save_dir:
            t0 = time.time()
            for source_slide in tqdm(source_list, ascii=True):
                t1 = time.time()
    
                source_ROIs = get_imagefiles(source_slide)
                total_ROIs = len(source_ROIs)
                left_ROIs = total_ROIs
                req_queue = mp.JoinableQueue(maxsize=left_ROIs)
                out_queue = mp.Queue(maxsize=left_ROIs)
                out_queue_flag1 = mp.Queue(maxsize=left_ROIs)
                out_queue_flag2 = mp.Queue(maxsize=left_ROIs)
                processes = []
                lk = mp.Lock()
###
# 20200518 for single process ver. from
                args=(req_queue, out_queue, out_queue_flag1, out_queue_flag2,
                                                         pwsi_gen, save_dir, lk, total_ROIs, msg, msg_fmt,
                                                         target_RM_H, target_W, column_flag)
                _ = calculate_source_matrices(*args)



#               # プロセスの初期駆動
#                for _ in range(max_workers):
#                    p = mp.Process(target=calculate_source_matrices,
#                                   args=(req_queue, out_queue, out_queue_flag1, out_queue_flag2,
#                                         pwsi_gen, save_dir, lk, total_ROIs, msg, msg_fmt,
#                                         target_RM_H, target_W, column_flag))
#                    p.start()
#                    #print("process %d start." % p.pid)
#                    processes.append(p)
#    
#                # request queueに全taskを積む
#                for _ in range(left_ROIs):
#                    req_queue.put(source_ROIs.pop())
#    
#                # request queueにpoison pillを積む
#                for _ in range(max_workers):
#                    req_queue.put(None)
#     
#                #task完了を待つ
#                req_queue.join()
#
# 20200518 for single process ver. to
###
                #append flag1 : few_cell_flag.csv
                if not out_queue_flag1.empty():
                    lflag1 = [out_queue_flag1.get() for _ in range(out_queue_flag1.qsize())]
                    df_flag1_add = pd.DataFrame(lflag1)
                    #save flag csvs.
                    if column_flag == 1:
                        df_flag1_add.to_csv(csv_path_vec1_6_thres4_sat25, index=False, mode='a', header=False)
            
                    elif column_flag == 2:
                        df_flag1_add.to_csv(csv_path_vec1_6_thres4_sat25, index=False, mode='a', header=False)

                #append flag2 : tissue_mask_exception_flag.csv
                if not out_queue_flag2.empty(): #if TissueMaskException occured.
                    lflag2 = [out_queue_flag2.get() for _ in range(out_queue_flag2.qsize())]
                    df_flag2_add = pd.DataFrame(lflag2)
                    df_flag2_add.to_csv(path_mask_exp_csv, index=False, mode='a', header=False)

                elapsed = time.time() - t1
                logger.info('\n')
                logger.info('[slide dir] {}'.format(source_slide))
                logger.info('[per slide] {:.4}s'.format(elapsed))

            #if TissueMaskException has not occured, remove csv.
            if os.path.exists(path_mask_exp_csv):
                if pd.read_csv(path_mask_exp_csv).empty:
                    os.remove(path_mask_exp_csv)

            elapsed = time.time() - t0
            logger.info('Finish! ( {:.4}s )'.format(elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('target',
                        help=('正規化に使う画像もしくは名前'
                              '（この画像の色合いに正規化する）'))
    parser.add_argument('outdir',
                        help='正規化に使ったパラメータの出力先ディレクトリ')
    parser.add_argument('image_source',
                        help='正規化したい画像の大元のディレクトリ')

    # オプション引数
    parser.add_argument('--source-list', nargs='+',
                        help='正規化したい画像のディレクトリのリスト')
    parser.add_argument('--save-dir', 
                        help='正規化後の画像を保存するディレクトリ')
    parser.add_argument('--target-W',
                        help='計算済みの正規化に使う画像の色空間パラメータ')
    parser.add_argument('--target-RMH', 
                        help='計算済みの正規化に使う画像の色濃度パラメータ')

    main(parser.parse_args())
