import fnmatch
import json
import os
import cv2
import _pickle as cPickle

train_images = {}


def load_images_resize(dataset_path="./training"):
    images_train = []
    # train_img = {}

    for path, dirs, files in os.walk(dataset_path):
        for file in files:
            if fnmatch.fnmatch(file, '*.jpg'):
                fullname = os.path.join(path, file)
                classe = os.path.basename(path)
                img = cv2.imread(fullname)
                height, weight, channels = img.shape
                # print(height, weight, channels)
                image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                images_train.append(image_gray)
                images_train.append(classe)
                print(images_train)
                # print(len(images_train))
                # train_images['image'] = image_gray
                # train_images['classe_name'] = classe
                # train_images.update({'image': image_gray, 'classe_name': classe})
                # print(train_images)

    return images_train, train_images


points_key = []


def detector_shift():
    global points_key
    # train_images = load_images_resize(dataset_path="./training")
    images_train = load_images_resize(dataset_path="./training")
    # print(images_train)

    # keypoints = {}
    filename = "keypoints.p"
    sift = cv2.xfeatures2d.SIFT_create()
    for image in images_train:
        keypoints_train, train_descriptor = sift.detectAndCompute(image, None)
        # print(keypoints_train)
        exit(0)
        points_key = []
        for key in keypoints_train:
            points_key.append((key.pt,
                               key.size,
                               key.angle,
                               key.response,
                               key.octave,
                               key.class_id
                               ))
        # cv2.imshow('Original image ', image)

        """with io.open('keypoints_train.txt', 'wb+', encoding='utf-8') as f:
            f.write(json.dumps(keypoints, ensure_ascii=False))"""
        keypoint = keypoints_train
        descriptor = train_descriptor
        keypoints = [keypoint, descriptor]
        # cPickle.dump(keypoints, open(keypoints_database, "wb+"))
        with open(filename, "wb+") as file:
            file.write(cPickle.dumps(points_key))

        image_with_keypoints = cv2.drawKeypoints(image, keypoints_train, image, color=(0, 255, 0),
                                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow('Image with keypoints and descriptors drawn ', image_with_keypoints)

        # print(points_key)
    return keypoints, points_key, train_descriptor


load_images_resize("training")
# detector_shift()
