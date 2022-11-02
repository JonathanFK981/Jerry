import os

import cv2

video_writer = cv2.VideoWriter("videos/Jerry81.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 640))
images_path = "datasets/Jerry/test1/81/81"
images_list = os.listdir(images_path)
images_list.sort()

for image_name in images_list:
    image = cv2.imread(os.path.join(images_path, image_name))
    video_writer.write(image)
    # show = cv2.resize(image, (1280, 720))
    # cv2.imshow("test", show)
    if cv2.waitKey(10) != ord('q'):
        pass
