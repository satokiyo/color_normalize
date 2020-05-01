import os
import pathlib
import cv2
import numpy as np
import pandas as pd

def output_H_geen_mask(source_ROI, norm_H, slide_name, origin_name, save_dir):

    vec_thres = 1.5

    # 各画像のpixel毎のヘマトキシリンのベクトルを取得(正規化後)
    hematoxylin_density_vec = norm_H[:,0]

    # 画像へreshape
    img_hematoxylin = hematoxylin_density_vec.reshape([375, 375])

    # 閾値以上は黒に、閾値以下は白にする(後に反転させるのでここでは逆)
    img_hematoxylin[img_hematoxylin >= vec_thres] = 255
    img_hematoxylin[img_hematoxylin < vec_thres] = 0
    # 要素がfloatなのでintへ書き換え
    img_hematoxylin = img_hematoxylin.astype(np.uint8)
    img_hematoxylin = cv2.bitwise_not(img_hematoxylin)
    # 二値化画像からRGB画像変換
    img_hematoxylin_mask = np.stack([img_hematoxylin]*3, axis=2)

    # 重ね合わせるソース画像を読み込んで重ね合わせる
    src_img = cv2.imread(source_ROI)
    dst = cv2.bitwise_and(src_img, img_hematoxylin_mask)

    # 黒の部分を緑に変換
    black = [0, 0, 0]
    green = [0, 255, 0]
    dst[np.where((dst == black).all(axis=2))] = green


    # 正規化後の画像を保存
    slide_dir = save_dir / origin_name / slide_name
    if not slide_dir.exists():
        slide_dir.mkdir(parents=True)

    new_source_ROI = pathlib.Path(source_ROI)
    path_to_save = str(slide_dir / new_source_ROI.name)
    
    cv2.imwrite(path_to_save, dst)


def output_H(source_ROI, norm_H, slide_name, origin_name, save_dir,
             out_queue_flag1, vec_thres, thres, sat_thres, flag_num, column_flag):
  
    # 各画像のpixel毎のヘマトキシリンのベクトルを取得(正規化後)
    hematoxylin_density_vec = norm_H[:,0]
    
    # ハードコピー
    hematoxylin_density_vec_hardcopy = np.copy(hematoxylin_density_vec)

    # 画像へreshape
    img_hematoxylin = hematoxylin_density_vec_hardcopy.reshape([375, 375])

    # 閾値以上は黒に、閾値以下は白にする(後に反転させるのでここでは逆)
    img_hematoxylin[img_hematoxylin >= vec_thres] = vec_thres
    img_hematoxylin[img_hematoxylin < vec_thres] = 0
    # 要素がfloatなのでintへ書き換え
    img_hematoxylin = img_hematoxylin.astype(np.uint8)

    m = cv2.countNonZero(img_hematoxylin)
    # h, w = img_hematoxylin.shape

    # saturation50より大きい値の面積を算出。
    # 元画像を読み込む
    src_img = cv2.imread(source_ROI)
    hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)

    hsvUpper = np.array([180,       255, 255])
    hsvLower = np.array([  0, sat_thres,   0])
    hsv_mask = cv2.inRange(hsv_img, hsvLower, hsvUpper)
    hm = cv2.countNonZero(hsv_mask)

    # マスクした部分の%(面積)を出す
    per = round(m/hm, 4)
    dec = int(bool(thres/100 > per))

    # decのbool値がTrueならflag_num,Falseなら0
    flag = flag_num if dec else 0

    # 正規化後の画像を保存
    slide_dir = save_dir / origin_name / slide_name
    if not slide_dir.exists():
        slide_dir.mkdir(parents=True)

    new_source_ROI = pathlib.Path(source_ROI)
    path_to_save = str(slide_dir / new_source_ROI.name)

    if column_flag == 1:
        out_queue_flag1.put([path_to_save, flag])

    elif column_flag == 2:
        out_queue_flag1.put([path_to_save, flag, per, m, hm])


def make_csv(save_dir, csv_name, column_flag,
             column1='image_path', column2='flag', column3='percent', column4='m',column5='hm'):
    """
    大元のフォルダ内ROIをすべてcsv出力すると非常に時間がかかるので、
    スライド毎にcsvを吐き出す仕様に変更する。
    """
    # csvPath設定
    # csv_path = os.path.join(str(slide_dir), csv_name) # スライド毎にcsv出力する場合
    str_save_dir = str(save_dir)
    up1_save_dir_l = str_save_dir.split('/')
    up1_save_dir_l.pop(-1)
    up1_save_dir_l.append("output")
    csv_path_dir = '/'.join(up1_save_dir_l)
    csv_path = os.path.join(csv_path_dir, csv_name) # 一つだけのスライドに処理をさせる場合

    if not os.path.exists(csv_path_dir):
        os.makedirs(csv_path_dir)
    if os.path.exists(csv_path): #remove existing csv
        os.remove(csv_path)
    if column_flag == 1: # 正規版
        csv_mat = pd.DataFrame(columns=[column1, column2])
        csv_mat.to_csv(csv_path, index=False)
    elif column_flag ==2: # debag用
        csv_mat = pd.DataFrame(columns=[column1, column2, column3, column4, column5])
        csv_mat.to_csv(csv_path, index=False)

    return csv_path
