from train_batch_kps import decoder_base
from train_batch_kps import encoder
from matplotlib import pyplot as plt
import tensorflow as tf 
from  PIL import Image
import numpy as np
import os 
import cv2
from timer import Timer

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

def visualize_heatmap(heatmap1, hw, i):
    h, w = hw
    heatmap1 = heatmap1/np.max(heatmap1)
    heatmap1 = heatmap1*255
    heatmap1 = cv2.resize(heatmap1, (w, h))
    heatmap1 = heatmap1.astype(np.uint8)
    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_HOT)
    print(heatmap1.shape)
    cv2.imwrite("heatmap{:d}.jpg".format(i),heatmap1)
    cv2.imshow('heatmap', heatmap1)
    cv2.waitKey(0)
class KeypointDetect():
    def __init__(self):
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        tfconfig.gpu_options.per_process_gpu_memory_fraction = 1.0
        g1 = tf.Graph()
        self.sess = tf.Session(config=tfconfig, graph=g1)
        with g1.as_default():
            self.images = tf.placeholder(tf.float32, shape = [1, 56, 56, 1])
            self.is_training = tf.placeholder(tf.bool)
            self.net = encoder(self.images,  self.is_training)
            self.logits = decoder_base(self.net,  self.is_training)
            self.saver = tf.train.Saver()
            self.sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            self.saver.restore(self.sess, "./output_with_crop/keypoints-55000")
    
    def predict_with_crop_img(self, img):
        prediction = self.sess.run(self.logits, feed_dict = {self.images:img, self.is_training: False})
        prediction = np.transpose(prediction, [0, 3, 1, 2])
        
       
        
        # keypoints prediction
        self.prediction = np.reshape(prediction, [21, -1])

        # keypoints transform
        pred_kps = np.argmax(self.prediction, axis = 1)
        pred_kps_y = pred_kps // 56
        pred_kps_x = pred_kps % 56
        pred_kps = np.hstack([pred_kps_x.reshape([21,-1]), pred_kps_y.reshape([21, -1])])

        return pred_kps
    def predict(self, imgs, boxes):
        
        #data preprocess
        xmin, ymin, xmax, ymax = boxes
        imgs = imgs[ymin:ymax, xmin:xmax]
        
        img_part = Image.fromarray(imgs)
        img_resized = np.array(img_part.resize((56,56)), dtype=np.float32)
        img_resized = img_resized - 103.9

        
        prediction = self.sess.run(self.logits, feed_dict = {self.images: img_resized.reshape([1, 56, 56, 1]), self.is_training: False})
        prediction = np.transpose(prediction, [0, 3, 1, 2])
        
        # keypoints prediction
        # for i in range(21):
        #     if i == 8 or i == 4:
        #         visualize_heatmap(prediction[0,i,:,:].copy(), (ymax-ymin, xmax-xmin), i)
        self.prediction = np.reshape(prediction, [21, -1])

        # keypoints transform
        pred_kps = np.argmax(self.prediction, axis = 1)
        pred_kps_y = pred_kps // 56
        pred_kps_x = pred_kps % 56
    
        pred_kps = np.hstack([pred_kps_x.reshape([21,-1]), pred_kps_y.reshape([21, -1])])
        pred_kps[:, 0] = pred_kps[:, 0] * (xmax - xmin) / 56 + xmin
        pred_kps[:, 1] = pred_kps[:, 1] * (ymax - ymin) / 56 + ymin
        pred_kps = pred_kps.astype(np.int32) 
        self.pred_kps = pred_kps
        return self.pred_kps
def visual_keypoints(im, pred_kps, boxes):
    # plt.imshow(im)
    x1, y1, x2, y2 = boxes
    # plt.gca().add_patch(plt.Rectangle((boxes[0], boxes[1]), boxes[2]-boxes[0], boxes[3]-boxes[1], fill = False, linewidth = 1))
    id = 0
    # cv2.imwrite("keypoint_origin.jpg", im[y1:y2, x1:x2])
    # cv2.rectangle(im, (x1,y1), (x2,y2), color =(0,255,0))
    for (x, y) in pred_kps:
        id += 1
        b = np.random.randint(low=0, high=255,size=1)[0]
        g = np.random.randint(low=0, high=255,size=1)[0]
        r = np.random.randint(low=0, high=255,size=1)[0]
        # print(b,g,r)
        cv2.circle(im, (x, y), 5, (int(b),int(g),int(r)),thickness=2)
        # plt.gca().add_patch(plt.Circle((x, y), 5, color = "r"))
        # plt.text(x, y, str(id))
    # cv2.imwrite("keypoint.jpg", im[y1:y2, x1:x2])
    cv2.imshow("pred_img", im)
    cv2.waitKey(0)
    # plt.show()

def visualize(im, pred_kps, labels):

    gt_kps_x1 = labels % 56
    gt_kps_y1 = labels // 56 
    gt_kps = np.hstack([gt_kps_x1.reshape(21, -1), gt_kps_y1.reshape(21, -1)])

    im_copy = im.copy()
    for i in range(21):
        
        # cv2.circle(im, (gt_kps[i, 0], gt_kps[i, 1]), 2, (0, 0, 0))
        cv2.circle(im, (pred_kps[i, 0], pred_kps[i, 1]), 1, (255, 0, 0))

    
    for i in range(21):
        cv2.circle(im_copy, (gt_kps[i, 0], gt_kps[i, 1]), 1, (0, 0, 0))
    im = cv2.resize(im+103.9, (224, 224)).astype(np.uint8)
    im_copy = cv2.resize(im_copy+103.9, (224, 224)).astype(np.uint8)
    cv2.imshow("pred_img", im)
    cv2.imshow("gt_img", im_copy)
    cv2.waitKey(0)
# sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
# # saver = tf.train.Saver()
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    keypoint_detect = KeypointDetect()
    # im_pre = './VOCdevkit/VOC2007/JPEGImages/{:s}.jpg'
    # kps_pre = './VOCdevkit/VOC2007/Keypoints/{:s}.txt'
    # img_index = open('./VOCdevkit/VOC2007/trainval.txt', 'r')
    im_pre = 'zebrafish_test/{:s}.jpg'
    kps_pre = 'zebrafish_test_keypoints/{:s}.txt'
    img_index = open('test_list.txt', 'r')
    results_kps = 'my_results/{:s}.txt'
    img_index = [i.strip() for i in img_index]
    for index in img_index:
        img = cv2.imread(im_pre.format(index))[:, :, 0]
        anno_parse = parse_rec_with_keypoint(kps_pre.format(index), im_pre.format(index))
        if anno_parse is not None:
            gt_boxes = anno_parse[0]['bbox']
            kps = keypoint_detect.predict(img, gt_boxes)
            np.savetxt(results_kps.format(index), kps)
            print(kps)
            # import pdb 
            # pdb.set_trace()
            visual_keypoints(img.copy(), kps.tolist(), gt_boxes)
        else:
            np.savetxt(results_kps.format(index), np.array([[1.0, 1.0]], dtype=np.int32))
    # from data import ZebrishData

    # zebrishdata_test = ZebrishData('train')
    # keypoint_detect = KeypointDetect()
    # test_timer = Timer()
    
    # # for i in range(100):
    # #     pred_kps = keypoint_detect.predict_with_crop_img(img)
    # for i in range(100):
    #     img, labels, _, gt_boxes = zebrishdata_test.get()
    #     test_timer.tic()
    #     pred_kps = keypoint_detect.predict_with_crop_img(img)
    #     test_timer.toc()
    #     # test_timer += e-s
    #     print('Average detecting time: {:.4f}s'.format(
    #         test_timer.average_time))
    #     visualize(img[0], pred_kps, labels[0])
    
    # print(test_timer / 1000)