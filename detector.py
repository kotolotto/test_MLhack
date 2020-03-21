import os
import cv2
import argparse
from glob import glob


def main(imgs_dir):
    eye_glasses_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye_tree_eyeglasses.xml')
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    imgs_filenames = sorted(glob(imgs_dir+'/*.jpg'))
    statuses = []
    for img_fname in imgs_filenames:
        status = detect_glasses(img_fname, face_cascade, eye_glasses_cascade)
        statuses.append(status)

    fname_w_glasses = [fname for fname, status in zip(imgs_filenames, statuses) if status]

    [print(fname) for fname in fname_w_glasses]


def detect_glasses(img_path, face_cascade, eye_glasses_cascade):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    is_glasses = []

    try:
        for (x, y, w, h) in faces:
            eyes_glasses = eye_glasses_cascade.detectMultiScale(gray[y:y + h, x:x + w])
            is_glasses.append(len(eyes_glasses) > 0)
    except:
        is_glasses = False

    return any(is_glasses)


if __name__ == "__main__":
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=dir_path)
    args = parser.parse_args()
    path = args.path if args.path is not None else os.getcwd()
    main(path)
