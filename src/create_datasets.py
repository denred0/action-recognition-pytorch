import os
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path
import shutil
import random

from tqdm import tqdm


def create_txt_files(root_directory, destination_directory, val_split_size, test_split_size):
    folder_names = os.listdir(root_directory)

    label_encoder = LabelEncoder()
    encoded = label_encoder.fit_transform(folder_names)

    # save labels dict to file
    with open(Path(destination_directory).joinpath('label_encoder.pkl'), 'wb') as le_dump_file:
        pickle.dump(label_encoder, le_dump_file)

    # save labels annotations in understandable format
    with open(Path(destination_directory).joinpath('labels_annotations.txt'), "w") as text_file:
        for id, label in enumerate(label_encoder.classes_):
            record = str(id) + ' ' + label
            print(record, file=text_file)

    all_data = []
    index = 0
    for folder in tqdm(folder_names):
        path = Path(root_directory).joinpath(folder)

        label = label_encoder.transform([folder]).item()

        for subdir, dirs, files in os.walk(path):
            for i, dir in enumerate(dirs):
                index += 1
                # create folder for video images
                video_folder = str(index) + '_video_label_' + str(label) + '_number_' + str(i)
                dest_path = Path(destination_directory).joinpath('videos').joinpath(video_folder)
                source_path = Path(root_directory).joinpath(folder).joinpath(dir)
                Path(dest_path).mkdir(parents=True, exist_ok=True)

                # filenames
                _, _, filenames = next(os.walk(source_path))

                for f_ind, file in enumerate(sorted(filenames)):
                    shutil.copy(os.path.join(source_path, file), os.path.join(dest_path, f"{(f_ind + 1):05d}.jpg"))

                record = [str(dest_path), 1, len(filenames), label]
                all_data.append(record)

        random.seed(4)
        random.shuffle(all_data)

        train_data = all_data[:int(round(len(all_data) * (1 - val_split_size), 0))]
        val_test_data = all_data[int(round(len(all_data) * (1 - val_split_size), 0)):]

        val_data = val_test_data[:int(round(len(val_test_data) * (1 - test_split_size), 0))]
        test_data = val_test_data[int(round(len(val_test_data) * (1 - test_split_size), 0)):]

        with open(Path(destination_directory).joinpath('train.txt'), "w") as text_file:
            for data in train_data:
                rec = str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3])
                print(rec, file=text_file)

        with open(Path(destination_directory).joinpath('val.txt'), "w") as text_file:
            for data in val_data:
                rec = str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + ' ' + str(data[3])
                print(rec, file=text_file)

        with open(Path(destination_directory).joinpath('test.txt'), "w") as text_file:
            for data in test_data:
                rec = str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2])
                print(rec, file=text_file)


if __name__ == '__main__':
    root_directory = 'data/initial_videos'
    destination_directory = 'dataset_dir'

    val_split_size = 0.3
    test_split_size = 0.5

    create_txt_files(root_directory, destination_directory, val_split_size, test_split_size)
