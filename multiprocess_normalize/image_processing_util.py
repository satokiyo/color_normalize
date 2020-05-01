from __future__ import division, print_function


import cv2
import numpy as np
import keras.backend as K

from PIL import Image
from keras.preprocessing import image as Kimage
from keras.applications.imagenet_utils import preprocess_input

def concat_h(im_list, blank_size=1, scale_ratio=1.0):
    """
    画像を横方向に連結
    """
    total_blank = sum(blank_size for _ in im_list) - blank_size

    scale_factor = scale_ratio ** -1

    height = max(im.height for im in im_list)
    height = int(height // scale_factor)
    total_width = int(sum(int(im.width // scale_factor) for im in im_list) + total_blank)

    dst_im = Image.new('RGB', (total_width, height))
    pos_x = 0
    for im in im_list:
        resized_im = im.resize((int(im.width // scale_factor) , int(im.height // scale_factor)))
        dst_im.paste(resized_im, (pos_x, 0))
        pos_x += int(blank_size + resized_im.width)

    return dst_im


def concat_v(im_list, blank_size=1, scale_ratio=1.0):
    """
    画像を縦方向に連結
    """
    total_blank  = sum(blank_size for _ in im_list) - blank_size

    scale_factor = scale_ratio ** -1

    width = max(im.width for im in im_list)
    width = int(width // scale_factor)

    total_height = int(sum(int(im.height // scale_factor) for im in im_list) + total_blank)
    dst_im = Image.new('RGB', (width, total_height))
    pos_y = 0
    for im in im_list:
        resized_im = im.resize((int(im.width // scale_factor) , int(im.height // scale_factor)))
        dst_im.paste(resized_im, (0, pos_y))
        pos_y += int(blank_size + resized_im.height)

    return dst_im


def bytes_to_RGB(bytes_data):
    """
    バイトデータをnumpy.ndarrayに変換
    """
    img_data = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_data = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    return img_data



PNG_ENCODE_OPTION = [int(cv2.IMWRITE_PNG_COMPRESSION), 100]
def encode_by_png(image_data):
    encoded_data = cv2.imencode('.png',
                                cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB),
                                PNG_ENCODE_OPTION)
    # [[...]]という形のリストが返ってくるので、インデックスが1のものを取得する
    return encoded_data[1].tostring()



PIL_INTERPOLATION_METHOD = {'nearest' : Image.NEAREST,
                            'bilinear': Image.BILINEAR,
                            'bicubic' : Image.BICUBIC,
                            'hamming' : Image.HAMMING,
                            'box'     : Image.BOX,
                            'lanczos' : Image.LANCZOS}
def transform_pil_image(pil_image, color_mode='rgb', target_size=None, interpolation='nearest'):
    """
    keras.preprocessing.imageモジュールのload_img関数から拝借したコード
    例外処理は行っていない
    """
    color_mode = color_mode.lower()
    mode = pil_image.mode
    if color_mode == 'grayscale':
        if mode != 'L':
            pil_image = pil_image.convert('L')

    elif color_mode == 'rgba':
        if mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')

    elif color_mode == 'rgb':
        if mode != 'rgb':
             pil_image = pil_image.convert('RGB')

    if target_size is not None:
        w_h_tuple = (target_size[1], target_size[0])
        if pil_image.size != w_h_tuple:
            resample  = PIL_INTERPOLATION_METHOD[interpolation]
            pil_image = pil_image.resize(w_h_tuple, resample)

    return pil_image


def vertical_flip(x):
    return Kimage.flip_axis(x, 0)

def horizontal_flip(x):
    return Kimage.flip_axis(x, 1)

def relative_flip(x):
    return vertical_flip(horizontal_flip(x))

def do_nothing(x):
    return x


# オリジナルのkeras_utils.pyから拝借した関数群 ###############################

def preprocess_input_tf(x, data_format=None, **kwargs):
    return preprocess_input(x, data_format, mode='tf')

def preprocess_input_caffe(x, data_format=None, **kwargs):
    return preprocess_input(x, data_format, mode='caffe')

def center_crop(x, center_crop_size):
    centerh, centerw = x.shape[0] // 2, x.shape[1] // 2
    lh, lw = center_crop_size[0] // 2, center_crop_size[1] // 2
    rh, rw = center_crop_size[0] - lh, center_crop_size[1] - lw
    h_start, h_end = centerh - lh, centerh + rh
    w_start, w_end = centerw - lw, centerw + rw
    return x[h_start:h_end, w_start:w_end, :]

def random_crop(x, random_crop_size):
    h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size[0]) // 2
    rangew = (w - random_crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)
    h_start, h_end = offseth, offseth + random_crop_size[0]
    w_start, w_end = offsetw, offsetw + random_crop_size[1]
    return x[h_start:h_end, w_start:w_end, :]



def random_transform(x,
                     dim_ordering='tf',
                     rotation_range=0.,
                     width_shift_range=0.,
                     height_shift_range=0.,
                     shear_range=0.,
                     zoom_range=0.,
                     channel_shift_range=0.,
                     fill_mode='nearest',
                     cval=0.,
                     horizontal_flip=False,
                     vertical_flip=False,
                     seed=None,
                     **kwargs):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
        # Returns
            A randomly transformed version of the input (same shape).
        """

        x = x.astype('float32')
        # x is a single image, so it doesn't have image number at index 0
        if dim_ordering == 'th':
            img_channel_axis = 0
            img_row_axis = 1
            img_col_axis = 2
        if dim_ordering == 'tf':
            img_channel_axis = 2
            img_row_axis = 0
            img_col_axis = 1

        if seed is not None:
            np.random.seed(seed)

        if np.isscalar(zoom_range):
            zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)

        # use composition of homographies
        # to generate final transform that needs to be applied
        if rotation_range:
            # theta = np.deg2rad(np.random.uniform(rotation_range, rotation_range))から変更
            theta = np.deg2rad(np.random.uniform(-rotation_range, rotation_range))
        else:
            theta = 0

        if height_shift_range:
            tx = np.random.uniform(-height_shift_range, height_shift_range)
            if height_shift_range < 1:
                tx *= x.shape[img_row_axis]
        else:
            tx = 0

        if width_shift_range:
            ty = np.random.uniform(-width_shift_range, width_shift_range)
            if width_shift_range < 1:
                ty *= x.shape[img_col_axis]
        else:
            ty = 0

        if shear_range:
            shear = np.deg2rad(np.random.uniform(shear_range, shear_range))
        else:
            shear = 0

        if zoom_range[0] == 1 and zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)

        transform_matrix = None
        if theta != 0:
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            transform_matrix = rotation_matrix

        if tx != 0 or ty != 0:
            shift_matrix = np.array([[1, 0, tx],
                                     [0, 1, ty],
                                     [0, 0, 1]])
            transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)

        if shear != 0:
            shear_matrix = np.array([[1, -np.sin(shear), 0],
                                    [0, np.cos(shear), 0],
                                    [0, 0, 1]])
            transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)

        if zx != 1 or zy != 1:
            zoom_matrix = np.array([[zx, 0, 0],
                                    [0, zy, 0],
                                    [0, 0, 1]])
            transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

        if transform_matrix is not None:
            h, w = x.shape[img_row_axis], x.shape[img_col_axis]
            transform_matrix = Kimage.transform_matrix_offset_center(transform_matrix, h, w)
            x = Kimage.apply_transform(x, transform_matrix, img_channel_axis,
                                      fill_mode=fill_mode, cval=cval)

        if channel_shift_range != 0:
            x = Kimage.random_channel_shift(x,
                                           channel_shift_range,
                                           img_channel_axis)
        if horizontal_flip:
            if np.random.random() < 0.5:
                x = Kimage.flip_axis(x, img_col_axis)

        if vertical_flip:
            if np.random.random() < 0.5:
                x = Kimage.flip_axis(x, img_row_axis)

        return x


