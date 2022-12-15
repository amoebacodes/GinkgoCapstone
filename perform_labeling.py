#%% load data

import os
import shutil
import itertools
import string
import numpy as np
import pandas as pd

protocol_df = pd.read_csv("autoprotocol_data.csv")
protocol_df.head()

# %%
imgs_dir = "images"
parent_dir = "labeled_data"

cols = list(range(1, 25))
rows = list(string.ascii_uppercase)[:16]
well_tups = list(itertools.product(rows, cols))
for idx, tup in enumerate(well_tups):
        well_tups[idx] = tup + (0,)
wells = [(tup[0] + str(tup[1]), tup[2]) for tup in well_tups]

# traverse images
for filename in os.listdir(imgs_dir):
    # make dir for each img inside labeled_dataset
    img_id = filename[14:-4]
    img_df = protocol_df.loc[(protocol_df['destination_plate_bcode'] == int(img_id))]
    plate_img_dir = os.path.join(parent_dir, img_id)
    if not os.path.exists(plate_img_dir):
        os.mkdir(plate_img_dir)
    # copy img over
    src = os.path.join(imgs_dir, filename)
    dst = os.path.join(plate_img_dir, filename)
    shutil.copyfile(src, dst)
    # get rows for this img
    labels_df = pd.DataFrame(wells, columns=['destination_well', 'label'])
    for _, row in img_df.iterrows():
        labels_df.loc[(labels_df['destination_well'] == row['destination_well']), ['label']] = 1
    # output csv to plate_img_dir
    csv_path = os.path.join(plate_img_dir, img_id + "_labels.csv")
    labels_df.to_csv(csv_path)
    
# %%
