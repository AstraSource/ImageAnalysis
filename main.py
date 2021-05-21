import image_augmentation as ia
# import neurowork as nw
import sys_helpers as sh
import sys
import os

WIDTH, HEIGHT = 200, 200
SOURCE_PATH = 'Resources\\Images_source'
OUTPUT_PATH = 'Resources\\Images_processed'


PARAM_UNLOAD = "unload"
PARAM_AUGMENT_GENERAL = "aug"
PARAM_AUGMENT_CROP = "crop"
PARAM_AUGMENT = ["rot", "shift", "rotshift", "noise", "all"]
PARAM_PROC_FULL = "full"


def main():
    # IMAGES PROCESSING
    source_path = sh.get_relative_path(SOURCE_PATH)
    output_path = sh.get_relative_path(OUTPUT_PATH)

    try:
        paths_i, labels_i = sh.collect_image_paths_and_labels(source_path)
    except StopIteration:
        print("Something is wrong with source folder")
    else:
        if PARAM_PROC_FULL not in sys.argv:
            if PARAM_UNLOAD in sys.argv:
                ia.unload_images(paths_i, labels_i, output_path)
                paths_o, labels_o = sh.collect_image_paths_and_labels(output_path)
                ia.del_small_images(WIDTH, HEIGHT, paths_o, labels_o)
            else:
                paths_o, labels_o = sh.collect_image_paths_and_labels(output_path)

            getparams = list(set(PARAM_AUGMENT) & set(sys.argv))
            if PARAM_AUGMENT_GENERAL in sys.argv:
                ia.augment_images(paths_o, labels_o, mode=getparams)

            if PARAM_AUGMENT_CROP in sys.argv:
                ia.crop_images(WIDTH, HEIGHT, paths_o)
        else:
            ia.unload_images(paths_i, labels_i, output_path)
            paths_o, labels_o = sh.collect_image_paths_and_labels(output_path)
            ia.del_small_images(WIDTH, HEIGHT, paths_o, labels_o)
            ia.augment_images(paths_o, labels_o, mode=["all"])
            ia.crop_images(WIDTH, HEIGHT, paths_o)

    # IMAGES ARE READY, BUILD THE MODEL.
    # nw.get_train_batches(paths_o, labels_o)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted!')
        try:
            sys.exit(1)
        except SystemExit:
            os.exit(1)
