import sys_helpers as sh
from PIL import Image
from PIL import ImageChops
from PIL import ImageFilter
from PIL import ImageColor
import random
import os
import numpy as np
import math
import tempfile


def crop_image(x_crop, y_crop, img_path):
    img = Image.open(img_path)
    x_size, y_size = img.size
    x_mid = x_size / 2
    y_mid = y_size / 2
    b1 = math.floor(max(x_mid - x_crop / 2, 0))
    b2 = math.floor(min(x_mid + x_crop / 2, x_size))
    b3 = math.floor(max(y_mid - y_crop / 2, 0))
    b4 = math.floor(min(y_mid + y_crop / 2, y_size))
    box = (b1, b3, b2, b4)
    img = img.crop(box)
    img.save(img_path)


def crop_images(x_crop, y_crop, image_paths):
    for img_path in image_paths:
        crop_image(x_crop, y_crop, img_path)


def del_small_images(x_small, y_small, image_paths, labels):
    for i in reversed(range(0, len(image_paths))):
        img_path = image_paths[i]
        img = Image.open(img_path)
        if x_small > img.size[0] or y_small > img.size[1]:
            img.close()
            os.remove(img_path)
            del image_paths[i]
            del labels[i]


def unload_images(image_paths, image_labels, output_path):
    sh.clean_directory(output_path)
    sh.create_directory(output_path)
    labels_unique = set(image_labels)
    for label in labels_unique:
        sh.clean_directory(os.path.join(output_path, str(label)))
        sh.create_directory(os.path.join(output_path, str(label)))

    for path, label in zip(image_paths, image_labels):
        img = Image.open(path)
        img.save(os.path.join(output_path, str(label), os.path.split(path)[1]))


def augment_images(image_paths, labels, mode):
    for i in reversed(range(0, len(image_paths))):
        path = image_paths[i]
        label = labels[i]
        img = Image.open(path)

        f, e = os.path.splitext(path)

        # Augment
        if "rot" in mode or "all" in mode:
            rot_img = aug1(img, (-15, 15))
            rot_img_path = f + '_rotated' + e
            rot_img.save(rot_img_path)
            image_paths.append(rot_img_path)
            labels.append(label)

        if "shift" in mode or "all" in mode:
            shift_img = aug2(img, 0.1)
            shift_img_path = f + '_shifted' + e
            shift_img.save(shift_img_path)
            image_paths.append(shift_img_path)
            labels.append(label)

        if "rotshift" in mode or "all" in mode:
            rot_shift_img = aug2(img, 0.1)
            rot_shift_img = aug1(rot_shift_img, (-15, 15))
            rot_shift_img_path = f + '_rotated_shifted' + e
            rot_shift_img.save(rot_shift_img_path)
            image_paths.append(rot_shift_img_path)
            labels.append(label)

        if "noise" in mode or "all" in mode:
            noise_img = aug3(img)
            noise_img_path = f + '_noise' + e
            noise_img.save(noise_img_path)
            image_paths.append(noise_img_path)
            labels.append(label)


# Rotate randomly
def aug1(img, conf):
    angle = random.randrange(conf[0], conf[1])
    mask = Image.new('1', img.size, 1)
    rotated = img.rotate(angle)
    mask = mask.rotate(angle)
    img = img.filter(filter=ImageFilter.GaussianBlur(radius=4))
    comp = Image.composite(rotated, img, mask)
    return comp


# Shift randomly
def aug2(img, rng):
    offset_x = math.floor(np.random.uniform(-rng, rng) * img.size[0])
    offset_y = math.floor(np.random.uniform(-rng, rng) * img.size[1])
    img = ImageChops.offset(img, offset_x, 0)

    if offset_x != 0:
        if offset_x > 0:
            line = offset_x
            resize_x = 0
        else:
            line = img.size[0] + offset_x - 1
            resize_x = line

        box = (line, 0, line + 1, img.size[1])
        sub_img = img.crop(box)
        sub_img = sub_img.resize((abs(offset_x), img.size[1]), Image.NEAREST)
        img.paste(sub_img, (resize_x, 0))

    img = ImageChops.offset(img, 0, offset_y)

    if offset_y != 0:
        if offset_y > 0:
            line = offset_y
            resize_y = 0
        else:
            line = img.size[1] + offset_y - 1
            resize_y = line

        box = (0, line, img.size[0], line + 1)
        sub_img = img.crop(box)
        sub_img = sub_img.resize((img.size[0], abs(offset_y)), Image.NEAREST)
        img.paste(sub_img, (0, resize_y))

    return img


def aug3(img):
    pil_map = Image.fromarray(np.random.randint(0, 255, (img.size[0], img.size[1], 3), dtype=np.dtype('uint8')))
    blend = Image.blend(img, pil_map, 0.2)
    return blend
