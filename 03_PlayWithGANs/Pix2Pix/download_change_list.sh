FILE=facades    # 400 images
FILE=night2day    # 20k images
FILE=maps    # 1096 images
FILE=cityscapes    # 2975 images
FILE=edges2shoes    # 50k images
FILE=edges2handbags    # 137k images

TARGET_DIR=./datasets/$FILE/
find "${TARGET_DIR}train" -type f -name "*.jpg" |sort -V > ./train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" |sort -V > ./val_list.txt

