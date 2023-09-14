import os
import os.path as osp
import sys
import torch
import random 
import torch.utils.data as data
import torch.nn.functional as F
import cv2
import numpy as np
from .config import cfg
from pycocotools import mask as maskUtils
import copy
from scipy.ndimage.morphology import distance_transform_edt
import math 
from pytorch_pretrained_bert import BertTokenizer, BertModel
from . import text_processing

def get_label_map():
    if cfg.dataset.label_map is None:
        return {x + 1: x + 1 for x in range(len(cfg.dataset.class_names))}
    else:
        return cfg.dataset.label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]) / scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class ReferDataset(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
        prep_crowds (bool): Whether or not to prepare crowds for the evaluation step.
    """

    def __init__(self, image_path, info_file, resize_gt=True,
                 img_size=None,
                 dataset='coco',   # coco or referit
                 augment=False,
                 has_gt=True):
        self.Anns = np.load(info_file, allow_pickle=True).item()['Anns']
        self.img_root = image_path
        self.img_list = os.listdir(image_path)
        self.img_size = img_size
        self.dataset = dataset
        self.has_gt = has_gt
        self.resize_gt = resize_gt

        self.lstm = cfg.lstm 
            
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.augment = augment 
        if self.augment:
            if self.dataset == 'coco':
                vocab_file = 'datasets/vocabulary_Gref.txt'
            else:
                vocab_file = 'datasets/vocabulary_referit.txt'
            self.vocab_dict = text_processing.load_vocab_dict_from_file(vocab_file)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, (target, masks, num_crowds)).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, text_batch, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, text_batch, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.Anns)

    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, masks, height, width, crowd).
                   target is the object returned by ``coco.loadAnns``.
            Note that if no crowd annotations exist, crowd will be None
        """
        if self.augment:
            augment_flip, augment_hsv, augment_affine = True,True,True
        else:
            augment_flip, augment_hsv, augment_affine = False,False,False
            
        temp_dict = self.Anns[index]
        file_name = temp_dict['file']
        text_batch = temp_dict['text_batch']
        # sent_batch: ['bottom right animal head'], use sent_batch[0] to get string.
        sent_batch = temp_dict['sent_batch'][0]
        bbox = copy.deepcopy(temp_dict['bbox'])
        # print(file_name, sent_batch)
        if bbox is not None:
            bbox[2], bbox[3] = bbox[0] + bbox[2], bbox[1] + bbox[3]  # xmin, ymin, xmax, ymax
        img = cv2.imread(osp.join(self.img_root + 'images/', file_name + '.jpg'), cv2.IMREAD_COLOR)
        
        mask = np.rint(
            cv2.imread(osp.join(self.img_root + 'mask/', file_name + '.png'), cv2.IMREAD_GRAYSCALE) / 255).astype(
            np.uint8)        
        
        height, width = mask.shape
        ### training
        if self.resize_gt:       
            ## random horizontal flip
            if augment_flip and random.random() > 0.5 and False:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
                bbox[0], bbox[2] = width-bbox[2]-1, width-bbox[0]-1
                sent_batch = sent_batch.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
                # cv2.imwrite('./visualize/flip_img.png', img)
                # cv2.imwrite('./visualize/flip_mask.png', mask*255)
                
            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
                # cv2.imwrite('./visualize/hsv.png', img)
            img, mask, ratio, dw, dh = letterbox(img, mask, self.img_size)
            if bbox is not None:
                bbox[0], bbox[2] = bbox[0] * ratio + dw, bbox[2] * ratio + dw
                bbox[1], bbox[3] = bbox[1] * ratio + dh, bbox[3] * ratio + dh
            ## random affine transformation
            if augment_affine:
                img, mask, bbox, M = random_affine(img, mask, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))

            #mask = mask_to_onehot(mask, 2)
            mask = np.reshape(mask, (-1, self.img_size, self.img_size))  
            edge_map = mask_to_binary_edges(mask, 2, num_classes = 1)
            masks = np.concatenate([mask, edge_map], axis=0)  # (2, 320, 320)
            if bbox is not None:
                target = [bbox[0], bbox[1], bbox[2], bbox[3], 1]
            else:
                target = [0,0,0,0, 1]
            target = np.array(target)
            #target = np.array(target) / self.img_size
            target = np.reshape(target, (1, -1))  # (5,) -> (1,5)
        else:  ### inference
            masks = mask 
            img, _, ratio, dw, dh = letterbox(img, None, self.img_size)
            # masks = np.reshape(masks, (-1, height, width))
            if bbox is not None:
                target = [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height, 0]
            else:
                target = [0,0,0,0,0]
            #target = [bbox[0], bbox[1], bbox[2], bbox[3], 1]
            target = np.array(target)
            target = np.reshape(target, (1, -1))

        # Separate out crowd annotations. These are annotations that signify a large crowd of
        # objects of said class, where there is no annotation for each individual object. Both
        # during testing and training, consider these crowds as neutral.
        num_crowds = 0

        if self.lstm:
            if self.resize_gt and self.augment:
                text_batch = text_processing.preprocess_sentence(sent_batch, self.vocab_dict, 20)
            else:
                text_batch = text_batch
            # word_mask = np.zeros(text_batch.shape)
        else:
            ## bert
            seq_len = len(text_batch)
            sent_batch = self.tokenizer.tokenize(sent_batch) # sent_batch is a list, use sent_batch[0] to get string
            
            # no more than 20-2 words, [CLS],[SEP]
            if len(sent_batch) > seq_len-2:
                sent_batch = sent_batch[0:(seq_len-2)]

            tokens = []
            tokens.append('[CLS]')
            tokens += sent_batch
            tokens.append('[SEP]')
            
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # word_mask = [1] * len(input_ids)

            # zero_pad up to ssequence length
            while len(input_ids) < seq_len:
                input_ids.append(0)
                # word_mask.append(0)
            
            assert len(input_ids) == seq_len
            # assert len(word_mask) == seq_len

            text_batch = input_ids

        return torch.from_numpy(img).permute(2, 0, 1), text_batch, target, masks, height, width, num_crowds

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        img_id = self.ids[index]
        path = self.coco.loadImgs(img_id)[0]['file_name']
        return cv2.imread(osp.join(self.root, path), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        return self.coco.loadAnns(ann_ids)


def enforce_size(img, word_idx, targets, masks, num_crowds, new_w, new_h):
    """ Ensures that the image is the given size without distorting aspect ratio. """
    with torch.no_grad():
        _, h, w = img.size()

        if h == new_h and w == new_w:
            return img, word_idx, targets, masks, num_crowds

        # Resize the image so that it fits within new_w, new_h
        w_prime = new_w
        h_prime = h * new_w / w

        if h_prime > new_h:
            w_prime *= new_h / h_prime
            h_prime = new_h

        w_prime = int(w_prime)
        h_prime = int(h_prime)

        # Do all the resizing
        img = F.interpolate(img.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        img.squeeze_(0)

        # Act like each object is a color channel
        masks = F.interpolate(masks.unsqueeze(0), (h_prime, w_prime), mode='bilinear', align_corners=False)
        masks.squeeze_(0)

        # Scale bounding boxes (this will put them in the top left corner in the case of padding)
        targets[:, [0, 2]] *= (w_prime / new_w)
        targets[:, [1, 3]] *= (h_prime / new_h)

        # Finally, pad everything to be the new_w, new_h
        pad_dims = (0, new_w - w_prime, 0, new_h - h_prime)
        img = F.pad(img, pad_dims, mode='constant', value=0)
        masks = F.pad(masks, pad_dims, mode='constant', value=0)

        return img, word_idx, targets, masks, num_crowds


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and (lists of annotations, masks)

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list<tensor>, list<tensor>, list<int>) annotations for a given image are stacked
                on 0 dim. The output gt is a tuple of annotations and masks.
    """
    targets = []
    word_idx = []
    imgs = []
    masks = []
    num_crowds = []

    for sample in batch:
        imgs.append(sample[0])
        word_idx.append(torch.FloatTensor(sample[1]).long())
        targets.append(torch.FloatTensor(sample[2][0]))
        masks.append(torch.FloatTensor(sample[2][1]))
        num_crowds.append(sample[2][2])

    return imgs, word_idx, (targets, masks, num_crowds)

# resize a rectangular image to a padded square
def letterbox(img, mask, height, color=(0.0, 0.0, 0.0)):  
    #color=(123.7, 116.3, 103.5)
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, (height, height), interpolation=cv2.INTER_AREA)  # resized, no border
    #img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    #img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, (height, height), interpolation=cv2.INTER_NEAREST)  # resized, no border
    img = img.astype(np.float32)
    if mask is not None:
        mask = mask.astype(np.float32)
    return img, mask, ratio, dw, dh

def random_affine(img, mask, targets, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(123.7, 116.3, 103.5), all_bbox=None):
    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    if mask is not None:
        maskw = cv2.warpPerspective(mask, M, dsize=(height, height), flags=cv2.INTER_NEAREST,
                                  borderValue=0)  # BGR order borderValue
    else:
        maskw = None

    return imw, maskw, targets, M

    # Return warped points also
    # if type(targets)==type([1]):
    #     targetlist=[]
    #     for bbox in targets:
    #         targetlist.append(wrap_points(bbox, M, height, a))
    #     return imw, maskw, targetlist, M
    # elif all_bbox is not None:
    #     targets = wrap_points(targets, M, height, a)
    #     for ii in range(all_bbox.shape[0]):
    #         all_bbox[ii,:] = wrap_points(all_bbox[ii,:], M, height, a)
    #     return imw, maskw, targets, all_bbox, M
    # elif targets is not None:   ## previous main
    #     targets = wrap_points(targets, M, height, a)
    #     return imw, maskw, targets, M
    # else:
    #     return imw

def wrap_points(targets, M, height, a):
    # n = targets.shape[0]
    # points = targets[:, 1:5].copy()

    # points = targets.clone()
    points = targets

    # area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
    area0 = (points[2] - points[0]) * (points[3] - points[1])

    # warp points
    xy = np.ones((4, 3))
    xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(1, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, 1).T

    # reject warped points outside of image
    np.clip(xy, 0, height, out=xy)
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    ## print(targets, xy)
    ## [ 56  36 108 210] [[ 47.80464857  15.6096533  106.30993434 196.71267693]]
    # targets = targets[i]
    # targets[:, 1:5] = xy[i]
    targets = xy[0]
    return targets   


def mask_to_binary_edges(mask, radius, num_classes):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    # mask==(mask>0)
    mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
        dist = dist[1:-1, 1:-1]  # delete padding border
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def mask_to_onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    _mask = [mask == (i + 1) for i in range(num_classes)]
    return np.array(_mask).astype(np.uint8)
