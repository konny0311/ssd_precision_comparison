import os


def extract_min_files(path, image_ids):
    cnt = 0
    with open(path) as f:
        for line in f:
            # クラス別テキストファイルに対応。
            splitted = line.strip().split(' ')
            if len(splitted) == 2:
                image_id = splitted[0]
                flag = splitted[1]
                if flag == '1' and not image_id in image_ids:
                    if image_id not in image_ids:
                        image_ids.append(image_id)
                        cnt += 1
            else:
                image_id = splitted[0]
                flag = splitted[2]
                if flag == '1' and not image_id in image_ids:
                    if image_id not in image_ids:
                        image_ids.append(image_id)
                        cnt += 1
            if cnt == CNTS[0]:
                break
    print(path, cnt)
    # return image_ids

def extract_files(path, image_ids):
    with open(path) as f:
        for line in f:
            # クラス別テキストファイルに対応。
            splitted = line.strip().split(' ')
            if len(splitted) == 2:
                image_id = splitted[0]
                flag = splitted[1]
                if flag == '1' and not image_id in image_ids:
                    if image_id not in image_ids:
                        image_ids.append(image_id)
            else:
                image_id = splitted[0]
                flag = splitted[2]
                if flag == '1' and not image_id in image_ids:
                    if image_id not in image_ids:
                        image_ids.append(image_id)
    # print(path, cnt)


def add_shortatge(path, less_cnt,image_ids):
    cnt = 0
    with open(path) as f:
        for line in f:
            image_id = line.strip().split(' ')[0]
            if image_id not in image_ids:
                image_ids.append(image_id)
                cnt += 1

            if cnt == less_cnt:
                break

CNTS = [120, 1000]
total = 4000
year = 2007

image_ids = []
base_dir = '/home/ubuntu/data/VOCdevkit/VOC{}/ImageSets/Main'.format(year)
# bus_train_path = os.path.join(base_dir, 'bus_train.txt')
# car_train_path = os.path.join(base_dir, 'car_train.txt')
# cat_train_path = os.path.join(base_dir, 'cat_train.txt')
# horse_train_path = os.path.join(base_dir, 'horse_train.txt')
# motorbike_train_path = os.path.join(base_dir, 'motorbike_train.txt')

# files = [os.path.join(base_dir, 'bus_val.txt'),
#             os.path.join(base_dir, 'car_val.txt'),
#             os.path.join(base_dir, 'cat_val.txt'),
#             os.path.join(base_dir, 'horse_val.txt'),
#             os.path.join(base_dir, 'motorbike_val.txt')]
# all_id_path = os.path.join(base_dir, '2007_valid_class_limitted.txt')

files = [os.path.join(base_dir, 'test.txt')]
        #    os.path.join(base_dir, 'car_trainval.txt'),
        #    os.path.join(base_dir, 'cat_trainval.txt'),
        #    os.path.join(base_dir, 'horse_trainval.txt'),
        #    os.path.join(base_dir, 'motorbike_trainval.txt')]
# all_id_path = os.path.join(base_dir, 'trainval_class_limitted.txt')

for file in files:
    extract_min_files(file, image_ids)

print(len(image_ids))

# less_cnt = CNTS[1] - len(image_ids)
# if less_cnt > 0:
#     add_shortatge(all_id_path, less_cnt, image_ids)
# print(len(image_ids))
with open('/home/ubuntu/data/VOCdevkit/VOC{}/ImageSets/Main/test_class_limitted.txt'.format(year), 'w') as f:
    for image_id in image_ids:
        f.write(image_id + '\n')
