import cv2
import os

obj_folder = '/home/user/storage/airlab/bop_data/ycbv/test_video'
video_folder = 'data/BOP/ycbv/ref_video_mask'
objs = os.listdir(obj_folder)
for obj in objs:
    video_name = os.path.join(video_folder, '{}.mp4'.format(obj))
    img_path = os.path.join(obj_folder, obj, 'mask')
    if not os.path.exists(img_path):
        continue
    images = [os.path.join(img_path, '{:06d}.jpg'.format(i)) for i in range(160)]
    frame = cv2.imread(os.path.join(img_path, images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    for image in images:
        out.write(cv2.imread(os.path.join(image)))

    cv2.destroyAllWindows()
    out.release()