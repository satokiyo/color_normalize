# !/bin/bash


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

############ IP : parameter ################################################################
# Target（パラメータファイル）があるディレクトリ（以降でW,RM_Hを指定しない場合はパラメータを保存するディレクトリ）
OUT_DIR="/mnt/c/Users/sato/root/prostate/20200518_color_normalize_single_process_check/color_normalize/ip/stain_normalization/375/target_matrices/P02212_5point"

# Targetの計算済みパラメータファイル
TARGET_W="/mnt/c/Users/sato/root/prostate/20200518_color_normalize_single_process_check/color_normalize/ip/stain_normalization/375/target_matrices/P02212_5point/W.npy" 
TARGET_RM_H="/mnt/c/Users/sato/root/prostate/20200518_color_normalize_single_process_check/color_normalize/ip/stain_normalization/375/target_matrices/P02212_5point/RM_H.npy"
# Target画像のディレクトリ（またはファイルパス）
TARGET_IMAGE="/mnt/c/Users/sato/root/prostate/20200518_color_normalize_single_process_check/color_normalize/ip/stain_normalization/P02212"


############ IP : source画像 ################################################################
# SOURCE画像格納ディレクトリ（FOLDER_ARRAYの更に上の階層のディレクトリを指定）
UPSTAGE_FOLDER_ARRAY="/mnt/c/Users/sato/root/prostate/20200518_color_normalize_single_process_check/color_normalize/ip/5p375RGB"


############ OUT : outputディレクトリ ##########################################################
# アウトプットのフォルダリスト作成
#UPSTAGE_OUTPUT_FOLDER_ARRAY="/media/prostate/20200414_prostate_normalize_ite_fix_lumithres0.8_out_shorttissue_flag/color_normalization_ite_fix_lumthres0.8_out_shorttissue_flag/v4_dai2ki_color_normalization_modified_version/out/20200414test/5p375N"
UPSTAGE_OUTPUT_FOLDER_ARRAY="/mnt/c/Users/sato/root/prostate/20200518_color_normalize_single_process_check/color_normalize/out/20200518test"

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#



find $UPSTAGE_FOLDER_ARRAY -type d
declare -a UPSTAGE_SOURCE_DIRS=($UPSTAGE_FOLDER_ARRAY)

# アウトプットのフォルダリスト作成
find $UPSTAGE_OUTPUT_FOLDER_ARRAY -type d
declare -a UPSTAGE_OUTPUT_SOURCE_DIRS=($UPSTAGE_OUTPUT_FOLDER_ARRAY)

for((i=0; i<${#UPSTAGE_SOURCE_DIRS[@]}; i++))
do

# 正規化したい画像が入っているディレクトリ一覧
FOLDER_ARRAY="${UPSTAGE_SOURCE_DIRS[i]}/*"
find $FOLDER_ARRAY -type d

# 正規化したい画像が入っているディレクトリ一覧
declare -a SOURCE_DIRS=($FOLDER_ARRAY)

# 画像ファイルの大元
IMAGE_SOURCE=${UPSTAGE_SOURCE_DIRS[i]}
# 正規化画像の保存先ディレクトリ
SAVE_DIR=${UPSTAGE_OUTPUT_SOURCE_DIRS[i]} 



######################### プログラム実行 ###################################################################

# パスを確認してください
python3 ./multiprocess_normalize/normalize_stain.py \
$TARGET_IMAGE $OUT_DIR $IMAGE_SOURCE \
--source-list "${SOURCE_DIRS[@]}" \
--save-dir $SAVE_DIR \
--target-W $TARGET_W \
--target-RMH $TARGET_RM_H

# 権限の開放
chmod -R 775 $OUT_DIR
chmod -R 775 $SAVE_DIR

done

##########################################################################################################

