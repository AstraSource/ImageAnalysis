import os
import shutil


def get_relative_path(d):
    directory_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(directory_path, d)


def collect_image_paths_and_labels(d):
    image_paths, labels = list(), list()
    label = 0
    classes = sorted(os.walk(d).__next__()[1])
    for c in classes:
        c_dir = os.path.join(d, c)
        walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                image_paths.append(os.path.join(c_dir, sample))
                labels.append(label)
        label = label + 1
    return image_paths, labels


def clean_directory(d):
    try:
        shutil.rmtree(d)
    except FileNotFoundError:
        print("Cannot find directory to delete", d)
    else:
        print("Deleted successfully", d)


def create_directory(d):
    try:
        os.mkdir(d)
    except OSError:
        print("Creation failed", d)
    else:
        print("Created successfully", d)
