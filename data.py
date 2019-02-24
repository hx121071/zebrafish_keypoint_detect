import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import config as cfg 
import xml.etree.ElementTree as ET
import pickle
from utils import get_keypoints_targets


# def parse_rec(filename):
#     """ Parse a PASCAL VOC xml file """
#     tree = ET.parse(filename)
#     objects = []
#     size = tree.find('size')
#     w = int(size.find('width').text)
#     h = int(size.find('height').text)

#     for obj in tree.findall('object'):
#         obj_struct = {}
#         obj_struct['name'] = obj.find('name').text
#         obj_struct['pose'] = obj.find('pose').text
#         obj_struct['truncated'] = int(obj.find('truncated').text)
#         obj_struct['difficult'] = int(obj.find('difficult').text)
#         bbox = obj.find('bndbox')
#         obj_struct['bbox'] = [int(bbox.find('xmin').text),
#                               int(bbox.find('ymin').text),
#                               int(bbox.find('xmax').text),
#                               int(bbox.find('ymax').text)]
#         obj_struct['width'] = w 
#         obj_struct['height'] = h
#         objects.append(obj_struct)

#     return objects
def parse_rec_with_keypoint(filename, im_path):
    objects = []
    # print(im_path)
    gt_keypoints = np.loadtxt(filename).astype(np.int32)
    obj_struct = {}
    # boxes
    if gt_keypoints.shape[0] == 22:
        gt_keypoints = gt_keypoints[1:, :] # 21 * 2
        x1 = np.min(gt_keypoints[:, 0])
        y1 = np.min(gt_keypoints[:, 1])
        x2 = np.max(gt_keypoints[:, 0])
        y2 = np.max(gt_keypoints[:, 1])

        im = cv2.imread(im_path)
        h,w,_ = im.shape
        dw1 = int((x2 - x1) * 0.15)
        dh1 = int((y2 - y1) * 0.15)
        x1 = max(x1 - dw1, 0)
        y1 = max(y1 - dh1, 0)
        x2 = min(x2 + dw1, w-1)
        y2 = min(y2 + dh1, h-1)
        
        boxes = [ x1, y1, x2, y2]
        obj_struct['difficult'] = 0
    else:
        return None
        # obj_struct['difficult'] = 1
        # boxes = [0, 0, 0, 0]
    obj_struct['bbox'] = boxes
    objects.append(obj_struct)
    # print(objects)
    return objects

class ZebrishData():

    def __init__(self, mode, rebuild=False):

        # need some path 
        self.data_path = './VOCdevkit/VOC2007'
        self.anno_path = os.path.join(self.data_path, 'Annotations')
        self.img_path = os.path.join(self.data_path, 'JPEGImages')
        self.kp_path = os.path.join(self.data_path, 'Keypoints')
        self.batch_size = cfg.batch_size
        if mode == 'train':
            self.im_index_list = os.path.join(self.data_path, 'trainval.txt')
        else:
            self.im_index_list = os.path.join(self.data_path, 'trainval.list')
            self.batch_size = 1
        
        self.image_size = cfg.image_size
        self.data = []
        self.rebuild = rebuild 
        self.cache_file = 'data_{:s}.pkl'.format(mode)
        self.epoch = 1 
        self.cursor = 0
        self.prepare()
        self.data_size = len(self.data)
    
    def random_crop(self, x1, y1, x2, y2, h, w):
        # width = np.random.randint(256, 295, size=1)[0]
        # height = np.random.randint(256, 320, size=1)[0]
        w1 = x2 - x1 
        h1 = y2 - y1
        dw1 = np.random.randint(-int(0.30*w1), int(0.30*w1), size=1)[0]
        dh1 = np.random.randint(-int(0.30*h1), int(0.30*h1), size=1)[0]
        # print(dw1, dh1)
        x1 += dw1
        y1 += dh1

        x2 += dw1
        y2 += dh1
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)
        return [x1, y1, x2, y2]
    def get(self):
        imgs = np.zeros((self.batch_size, cfg.image_size, cfg.image_size, 1), dtype=np.float32)
        labels = np.zeros((self.batch_size, cfg.keypoints_num), dtype=np.int32)
        labels_weights = np.zeros_like(labels, dtype=np.float32)
        count = 0

        gt_boxes_list = []
        kps_list = []
        # print(self.batch_size)
        while count < self.batch_size:

            im_path = self.data[self.cursor]['im_path']
            gt_boxes = self.data[self.cursor]['gt_boxes']
            kps = self.data[self.cursor]['im_kp']
            img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            x1, y1, x2, y2 = gt_boxes
            random_crop_flag = np.random.uniform(size=1)[0]
            if random_crop_flag>1.0:
                gt_boxes = self.random_crop(x1, y1, x2, y2, h, w)
            imgs[count] = self.read_img(im_path, gt_boxes)
            gt_boxes_list.append(gt_boxes)
            kps_list.append(self.read_keypoints(kps))
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.data):
                np.random.shuffle(self.data)
                self.cursor = 0
                self.epoch += 1
            
        gt_boxes_np = np.stack(gt_boxes_list)
        kps_np = np.stack(kps_list)
        labels,labels_weights = get_keypoints_targets(gt_boxes_np, kps_np)
        # print(gt_boxes_np.shape)
        # print(kps_np.shape)
        return imgs, labels, labels_weights
        # return imgs, labels, labels_weights, gt_boxes_list


    def read_keypoints(self, kp_path):

        kps = np.loadtxt(kp_path, dtype=np.float32)[1:]

        return kps 
    def read_img(self, img_path, proposal):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        x1, y1, x2, y2 = proposal
        # print(x1, x2, y1, y2) 
        img = img[y1:y2, x1:x2]

        # for visualize
        # origin_image = img.copy()
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img = cv2.resize(img, (cfg.image_size, cfg.image_size)).astype(np.float32)
        # cv2.imshow("test", img)
        # cv2.waitKey(0)
        img = img - 103.9
        img = np.reshape(img, (img.shape[0], img.shape[1], -1))
        # cv2.imshow("test1", img)
        # cv2.waitKey(0)
        return img
    def prepare(self):
        
        if os.path.isfile(self.cache_file) and not self.rebuild:
            print('Loading data from: ', self.cache_file)
            with open(self.cache_file, 'rb') as f:
                 self.data = pickle.load(f)
                 return
        
        print('Processing data from: ', self.data_path)
        im_index_list = np.loadtxt(self.im_index_list, dtype=str)
        # print(im_index_list)

        for im_index in im_index_list:
            single_data = {}
            im_anno = self.anno_path + '/' + im_index + '.xml'
            im_path = self.img_path + '/' + im_index + '.jpg'
            im_kp = self.kp_path + '/' + im_index + '.txt'

            # anno_parse = parse_rec(im_anno)[0]
            anno_parse = parse_rec_with_keypoint(im_kp, im_path)
            # kps = np.loadtxt(im_kp, dtype=np.int32)
            if anno_parse is not None:
            # diff = anno_parse['difficult']
            # if not diff and kps.shape[0] == 22:
                anno_parse = anno_parse[0]
                gt_boxes = anno_parse['bbox']
                # w = anno_parse['width']
                # h = anno_parse['height']
                # gt_boxes[0] = max(gt_boxes[0] - 20, 0)
                # gt_boxes[1] = max(gt_boxes[1] - 20, 0)
                # gt_boxes[2] = min(gt_boxes[2] + 20, w-1)
                # gt_boxes[3] = min(gt_boxes[3] + 20, h-1)


                single_data['gt_boxes'] = gt_boxes 
                single_data['im_path'] = im_path 
                single_data['im_kp'] = im_kp

                self.data.append(single_data)
        np.random.shuffle(self.data)

        print('Saving data to: ', self.cache_file)
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.data, f)
        print('Done prepare data')
        return

def visualize_origin_data(data):
    """
    data is list of dict which has keys gt_boxes, im_path, im_kp 
    """

    for item in data:
        gt_boxes = item['gt_boxes']
        im_path = item['im_path']
        im_kp = item['im_kp']

        im = cv2.imread(im_path)
        
        print(im_path)
        kps = np.loadtxt(im_kp, dtype=np.int32)
        print("kps is: ", kps)
        assert(kps.shape[0] == 22)

        x1, y1, x2, y2 = gt_boxes
        proposal = im[y1:y2, x1:x2, :].copy()
        h, w, _ = proposal.shape
        proposal = cv2.resize(proposal, (56, 56))
        cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0))
        for i in range(1, 22):
            kp_x1, kp_y1 = kps[i]
            cv2.circle(im, (kp_x1, kp_y1), 5, (0, 0, 255))

        cv2.imshow("test", im)

        
       
        # trans_kps = []
        for i in range(1, 22):
            kp_x1, kp_y1 = kps[i]
            trans_x1 = int((kp_x1 - x1) * cfg.image_size / w) 
            trans_y1 = int((kp_y1 - y1) * cfg.image_size / h)
            cv2.circle(proposal, (trans_x1, trans_y1), 2, (0, 0, 255))
        cv2.imshow("resize_test", proposal) 
        cv2.waitKey(0)


def visualize_batch_data(imgs, labels, labels_weights):
    print(labels_weights)
    print(labels_weights.shape)
    for i in range(cfg.batch_size):
        img = imgs[i] 
        label = labels[i] 
        label_weights = labels_weights[i] 
        
        for j in range(cfg.keypoints_num):
            if label_weights[j] > 0:
                
                kps_x1 = label[j] % cfg.image_size
                kps_y1 = label[j] // cfg.image_size
                cv2.circle(img, (kps_x1, kps_y1), 2, (255, 0, 0))

        cv2.imshow("test", img) 
        cv2.waitKey(0)        

if __name__ == "__main__":
    # show_data()
    # gen_label()
    zebrishdata = ZebrishData('train')
    print(zebrishdata.data_size)
    # visualize(zebrishdata.data)
    imgs, labels, labels_weights = zebrishdata.get()
    visualize_batch_data(imgs, labels, labels_weights)
    # visualize_origin_data(zebrishdata.data)
    # import pdb 
    # pdb.set_trace()
