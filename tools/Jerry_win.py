import os
import numpy as np
import json
import cv2


# Use the same script for MOT16
DATA_PATH = '/home/administrator/deeplearning/code/Gayhub/detection/MOT/Jerry/ByteTrack/datasets/Jerry_v1'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')#E:/GayHub/data/NN/ByteTrack/datasets/Jerry/annotations
# SPLITS = ['train_half', 'val_half', 'train', 'test']  # --> split training data to train_half and val_half.
SPLITS = ['train_half', 'val_half']
HALF_VIDEO = True#训练集分一半为验证集
CREATE_SPLITTED_ANN = True#
CREATE_SPLITTED_DET = True#


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)
    '''
    测试集标注
    '''
    for split in SPLITS:
        if split == "test":
            data_path = os.path.join(DATA_PATH, 'test') #datasets/Jerry/test   
        else:
            data_path = os.path.join(DATA_PATH, 'train') #datasets/Jerry/train
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))#datasets/Jerry/annotations/train、test、train_half、val_half.json  
        out =  {'images': [], 
                'annotations': [], 
                'videos': [],
                'categories':[{'id': 1, 'name': 'sailing_boat'},
                              {'id': 2, 'name': 'fishing_boat'},
                              {'id': 3, 'name': 'floater'},
                              {'id': 4, 'name': 'passenger_ship'},
                              {'id': 5, 'name': 'speedboat'},
                              {'id': 6, 'name': 'cargo'},
                              {'id': 7, 'name': 'special_ship'}]}
        seqs = os.listdir(data_path)#(4、12、14、15等19个文件夹)
        image_cnt = 0#图片计数器
        ann_cnt = 0#标注计数器
        video_cnt = 0#录像计数器，即在训练/测试集中累计每一个文件夹的计数器，统计有多少个子文件夹
        tid_curr = 0#
        tid_last = -1#用于在下一个视频序列时，ID数接着上一个视频序列最大值
        for seq in sorted(seqs):# 1（录像计数器循环）
            if '.DS_Store' in seq:
                continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_path = os.path.join(data_path, seq)#datasets/Jerry/train/4
            img_path = os.path.join(seq_path, seq)#datasets/Jerry/train/4/4
            ann_path = os.path.join(seq_path, 'gt/gt.txt')#datasets/Jerry/train/4/gt/gt.txt
            images = os.listdir(img_path)#(4文件夹下的所有图片)
            num_images = len([image for image in images if 'jpg' in image])  # half and half  4文件夹下的图片个数：300

            if HALF_VIDEO and ('half' in split):#取一半为真且split字符串含half，前一半为train_half个数，后一半为val_half个数
                image_range = [0, num_images // 2] if 'train' in split else \
                              [num_images // 2 + 1, num_images - 1]
            else:#train和test个数
                image_range = [0, num_images - 1]

            for i in range(num_images):#图片计数器循环
                if i < image_range[0] or i > image_range[1]:#图片数量小于0或者大于最大个数
                    continue
                img = cv2.imread(os.path.join(data_path, '{}/{}/out{}_{:04d}.jpg'.format(seq, seq, seq, i + 1)))
                height, width = img.shape[:2]
                image_info = {'file_name': '{}/{}/out{}_{:04d}.jpg'.format(seq, seq, seq, i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.图片序号
                              'frame_id': i + 1 - image_range[0],  # image number in the video sequence, starting from 1.帧序号
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.第1张图片的前一个图片序号为-1
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,#最后1张图片的后一张图片序号为-1，299+2=301>300
                              'video_id': video_cnt,#当前录像计数器序号（4、12等19个录像）
                              'height': height, 'width': width}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))#1：300 images
            '''
            训练集标注
            '''
            if split != 'test':
                det_path = os.path.join(seq_path, 'det/det.txt')#datasets/Jerry/train/4/4/det/det.txt
                anns = np.loadtxt(ann_path, dtype=np.float32, delimiter=',')#datasets/Jerry/train/4/4/gt/gt.txt
                dets = np.loadtxt(det_path, dtype=np.float32, delimiter=',')#datasets/Jerry/train/4/4/det/det.txt
                if CREATE_SPLITTED_ANN and ('half' in split):#训练集和验证集分离的标注
                    anns_out = np.array([anns[i] for i in range(anns.shape[0])
                                         if int(anns[i][0]) - 1 >= image_range[0] and
                                         int(anns[i][0]) - 1 <= image_range[1]], np.float32)
                    anns_out[:, 0] -= image_range[0]
                    gt_out = os.path.join(seq_path, 'gt/gt_{}.txt'.format(split))
                    fout = open(gt_out, 'w')
                    for o in anns_out:
                        fout.write('{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), int(o[2]), int(o[3]), int(o[4]), int(o[5]),
                                    int(o[6]), int(o[7]), o[8]))
                    fout.close()
                if CREATE_SPLITTED_DET and ('half' in split):
                    dets_out = np.array([dets[i] for i in range(dets.shape[0])
                                         if int(dets[i][0]) - 1 >= image_range[0] and
                                         int(dets[i][0]) - 1 <= image_range[1]], np.float32)
                    dets_out[:, 0] -= image_range[0]
                    det_out = os.path.join(seq_path, 'det/det_{}.txt'.format(split))
                    dout = open(det_out, 'w')
                    for o in dets_out:
                        dout.write('{:d},{:d},{:.1f},{:.1f},{:.1f},{:.1f},{:.6f}\n'.format(
                                    int(o[0]), int(o[1]), float(o[2]), float(o[3]), float(o[4]), float(o[5]),
                                    float(o[6])))
                    dout.close()

                print('{} ann images'.format(int(anns[:, 0].max())))
                
                for i in range(anns.shape[0]):
                    frame_id = int(anns[i][0])#帧号
                    if frame_id - 1 < image_range[0] or frame_id - 1 > image_range[1]:
                        continue
                    track_id = int(anns[i][1])#轨迹号
                    cat_id = int(anns[i][7])#类别号
                    ann_cnt += 1
                    if (anns[i][8] == 2):  # visibility.可见性
                        continue
                    # if not ('15' in DATA_PATH):
                        #if not (float(anns[i][8]) >= 0.25):  # visibility.
                            #continue
                        # if not (int(anns[i][6]) == 1):  # whether ignore.置信度不为1的忽略掉
                        #     continue
                        # if int(anns[i][7]) in [3, 4, 5, 6, 9, 10, 11]:  # Non-person种类不是人的类别忽略掉
                        #     continue
                        # if int(anns[i][7]) in [2, 7, 8, 12]:  # Ignored person
                        #     #category_id = -1
                        #     continue
                        # else:
                        #     category_id = 1  # pedestrian(non-static)
                                # if not track_id == tid_last:
                                #     tid_curr += 1
                                #     tid_last = track_id
                    # else:
                    #     category_id = 1
                    ann = {'id': ann_cnt,
                           'category_id': cat_id,
                           'image_id': image_cnt + frame_id,
                           'track_id': track_id,
                           'bbox': anns[i][2:6].tolist(),
                           'conf': float(anns[i][6]),
                           'iscrowd': 0,
                           'area': float(anns[i][4] * anns[i][5])}
                    out['annotations'].append(ann)
            image_cnt += num_images
            # print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
