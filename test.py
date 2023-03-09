# import networks

# from keras import layers

import os

# Rename the files!!
import os
 
# os.chdir('./datasets/cuhk_dataset/cuhk_photos')
# print(os.getcwd())
 
# for count, f in enumerate(os.listdir()):
#     f_name, f_ext = os.path.splitext(f)
#     f_name = "geek" + str(count)
 
#     new_name = f'{f_name}{f_ext}'
#     os.rename(f, new_name)

photo_files =  sorted(os.listdir('./datasets/cuhk_dataset/photos'))
sketch_files =  sorted(os.listdir('./datasets/cuhk_dataset/sketches'))

N = len(photo_files)

### Create an evaluation dataset at random, with 30 images
nb_imgs = 30
from random import sample
random_indices = sample(range(0, N), nb_imgs)
print(random_indices)
print(N)
from shutil import copy
for i in random_indices:
    photo_filename = photo_files[i]
    sketch_filename = sketch_files[i]
    print(photo_filename, sketch_filename)
    photo_dir = './datasets/cuhk_dataset/photos'
    sketch_dir = './datasets/cuhk_dataset/sketches'
    photo_fp = os.path.join(photo_dir,photo_filename)
    sketch_fp = os.path.join(sketch_dir, sketch_filename)
    # print(photo_fp, sketch_fp)
    photo_targetdir = './datasets/evaluation_dataset/photos'
    sketch_targetdir = './datasets/evaluation_dataset/sketches'
    copy(photo_fp, photo_targetdir)
    copy(sketch_fp, sketch_targetdir)
    # print('\n')
# 


# new_photo_ids = []
# new_sketch_ids = []

# for photo_file, sketch_file in zip(photo_files, sketch_files):
#     photo_id = photo_file[:-4].lower()
#     sketch_id = sketch_file[:-8].lower()
#     same = sketch_id == photo_id
#     checkbox = '[V]' if same else '[ ]'

#     photo_split = photo_id.split('-')
#     sketch_split = sketch_id.split('-')

#     gd_photo = photo_split[0]
#     gd_sketch = sketch_split[0]
#     photo_nb = ''.join(photo_split[1:])
#     sketch_nb = ''.join(sketch_split[1:])

#     # New ids without the extra digit in the gender code
#     new_photo_id = '-'.join([gd_photo[0],photo_nb[:3],photo_nb[3:]])
#     new_sketch_id = '-'.join([gd_sketch[0],sketch_nb[:3],sketch_nb[3:]])

#     new_photo_ids.append(photo_id)
#     new_sketch_ids.append(sketch_id)

#     # if not same:
#     #     same = gd_photo[0] == gd_sketch[0] and photo_nb==sketch_nb
#     #     checkbox='[L]' if same else '[ ]'
#     print(f'{checkbox} - Photo({photo_id}) v. Sketch({sketch_id})')

# print(f'Unique (new) photo IDs: {len(set(new_photo_ids))}/{len(photo_files)}')
# print(f'Unique (new) sketch IDs: {len(set(new_sketch_ids))}/{len(sketch_files)}')








# script = "function y = myScript(x)\n" \
#          "    y = x-5" \
#          "end"

# with open("myScript.m","w+") as f:
#     f.write(script)

# oc.myScript(7)


####### DEEP ##################
# intermediate_dim = 56
# img_dim = 96*96*3

# deep_encoder = keras.Sequential([
#     layers.Dense(256, activation='relu', input_shape=(None,img_dim)),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(64, activation='relu')
# ], name='deep_encoder')
# deep_encoder.summary()

# deep_decoder = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(None,intermediate_dim)),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(512, activation='relu'),
#     layers.Dense(1024, activation='relu'),
#     layers.Dense(img_dim, activation='sigmoid')
# ], 'deep_decoder')
# deep_decoder.summary()

# deep_vae
