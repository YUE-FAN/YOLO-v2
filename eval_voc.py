# encoding:utf-8
#
# created by xiongzihua
#
import os
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from src.yolo_net import Yolo
from torch.autograd import Variable
from src.utils import post_processing

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def voc_ap(rec, prec, use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(preds, target, VOC_CLASSES=VOC_CLASSES, threshold=0.5, use_07_metric=False, ):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for i, class_ in enumerate(VOC_CLASSES):
        pred = preds[class_]  # [[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0:  # 如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_, ap))
            aps += [ap]
            break
        # print(pred)
        image_ids = [x[0] for x in pred]
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        npos = 0.
        for (key1, key2) in target:
            if key2 == class_:
                npos += len(target[(key1, key2)])  # 统计这个类别的正样本，在这里统计才不会遗漏
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d, image_id in enumerate(image_ids):
            bb = BB[d]  # 预测框
            if (image_id, class_) in target:
                BBGT = target[(image_id, class_)]  # [[],]
                for bbgt in BBGT:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (bbgt[2] - bbgt[0] + 1.) * (
                                bbgt[3] - bbgt[1] + 1.) - inters
                    if union == 0:
                        print(bb, bbgt)

                    overlaps = inters / union
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt)  # 这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id, class_)]  # 删除没有box的键值
                        break
                fp[d] = 1 - tp[d]
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print(rec,prec)
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_, ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))


def test_eval():
    preds = {
        'cat': [['image01', 0.9, 20, 20, 40, 40], ['image01', 0.8, 20, 20, 50, 50], ['image02', 0.8, 30, 30, 50, 50]],
        'dog': [['image01', 0.78, 60, 60, 90, 90]]}
    target = {('image01', 'cat'): [[20, 20, 41, 41]], ('image01', 'dog'): [[60, 60, 91, 91]],
              ('image02', 'cat'): [[30, 30, 51, 51]]}
    voc_eval(preds, target, VOC_CLASSES=['cat', 'dog'])


if __name__ == '__main__':
    # test_eval()
    image_size = 448
    conf_threshold = 0.35  # TODO: make sure this is the correct threshold
    nms_threshold = 0.5

    from collections import defaultdict
    from tqdm import tqdm

    target = defaultdict(list)
    preds = defaultdict(list)
    image_list = []  # image path list

    f = open('voc2007test.txt')
    lines = f.readlines()
    file_list = []
    for line in lines:
        splited = line.strip().split()
        file_list.append(splited)
    f.close()
    print('---prepare target---')
    for index, image_file in enumerate(file_list):
        image_id = image_file[0]

        image_list.append(image_id)
        num_obj = (len(image_file) - 1) // 5
        for i in range(num_obj):
            x1 = int(image_file[1 + 5 * i])
            y1 = int(image_file[2 + 5 * i])
            x2 = int(image_file[3 + 5 * i])
            y2 = int(image_file[4 + 5 * i])
            c = int(image_file[5 + 5 * i])
            class_name = VOC_CLASSES[c]
            target[(image_id, class_name)].append([x1, y1, x2, y2])

    # start test
    print('---start test---')
    model = Yolo(20)
    model.load_state_dict(torch.load('./trained_models/only_params_trained_yolo_voc'))
    model.eval()
    model.cuda()
    with torch.no_grad():
        count = 0
        for image_path in tqdm(image_list):

            image = cv2.imread(os.path.join('/BS/database11/allimgs/', image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            image = cv2.resize(image, (image_size, image_size))
            image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
            image = image[None, :, :, :]
            width_ratio = float(image_size) / width
            height_ratio = float(image_size) / height
            data = Variable(torch.FloatTensor(image))
            if torch.cuda.is_available():
                data = data.cuda()
            with torch.no_grad():
                logits = model(data)  # 14x14x30
                predictions = post_processing(logits, image_size, VOC_CLASSES, model.anchors, conf_threshold,
                                              nms_threshold)
            if len(predictions) == 0:
                continue
            else:
                predictions = predictions[0]
            output_image = cv2.imread(image_path)
            for pred in predictions:  # pred is [x1, y1 ,w , h, prob, cls], e.g. [15, 25, 200, 102, 0.66, 'cat']
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
                ymax = int(min((pred[1] + pred[3]) / height_ratio, height))

                preds[pred[5]].append([image_path, pred[4], xmin, ymin, xmax, ymax])

        print('---start evaluate---')
        voc_eval(preds, target, VOC_CLASSES=VOC_CLASSES)