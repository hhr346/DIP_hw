FILE=facades    # 400 images
FILE=cityscapes    # 2975 images
FILE=maps    # 1096 images
FILE=edges2handbags    # 137k images
FILE=night2day    # 20k images
FILE=edges2shoes    # 50k images

URL=http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/$FILE.tar.gz
TAR_FILE=./datasets/$FILE.tar.gz
TARGET_DIR=./datasets/$FILE/
mkdir -p $TARGET_DIR
echo "Downloading $URL dataset..." to $TARGET_DIR
wget -N $URL -O $TAR_FILE
mkdir -p $TARGET_DIR
tar -zxvf $TAR_FILE -C ./datasets/
rm $TAR_FILE

find "${TARGET_DIR}train" -type f -name "*.jpg" |sort -V > ./train_list.txt
find "${TARGET_DIR}val" -type f -name "*.jpg" |sort -V > ./val_list.txt
