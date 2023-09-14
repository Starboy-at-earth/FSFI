from locale import normalize
import os
os.environ["CUDA_VISIBLE_DEVICES"]= '0'
import cv2
from data import ReferDataset, get_label_map, MEANS  # , COLORS
from utils.functions import MovingAverage, ProgressBar
from utils import timer
from utils.functions import SavePath
from data import cfg, set_cfg, set_dataset
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
# from my_model import model
# from model3 import model
# from model5_restnet50 import model
# from model8_gate import model
# from sample_0114 import model
from model import model
# from model5_resnet50_sample import model
import time
from tqdm import tqdm

def seed_torch(seed=666):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
seed_torch()


# # all boxes are [num, height, width] binary array
def compute_mask_IU(masks, target):
    assert (target.shape[-2:] == masks.shape[-2:])
    I = np.sum(np.logical_and(masks, target))
    U = np.sum(np.logical_or(masks, target))
    return I, U


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='YOLACT COCO Evaluation')
    parser.add_argument('--trained_model',
                        default='weights/Base_8_90000.pth', type=str,
                        help='Trained state_dict file path to open. If "interrupt", this will open the interrupt file.')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to evaulate model')
    parser.add_argument('--fast_nms', default=True, type=str2bool,
                        help='Whether to use a faster, but not entirely correct version of NMS.')
    parser.add_argument('--cross_class_nms', default=False, type=str2bool,
                        help='Whether compute NMS cross-class or per-class.')
    parser.add_argument('--display_masks', default=True, type=str2bool,
                        help='Whether or not to display masks over bounding boxes')
    parser.add_argument('--display_bboxes', default=True, type=str2bool,
                        help='Whether or not to display bboxes around masks')
    parser.add_argument('--display_text', default=True, type=str2bool,
                        help='Whether or not to display text (class [score])')
    parser.add_argument('--display_scores', default=True, type=str2bool,
                        help='Whether or not to display scores in addition to classes')
    parser.add_argument('--display', dest='display', action='store_true',
                        help='Display qualitative results instead of quantitative ones.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='Shuffles the images when displaying them. Doesn\'t have much of an effect when display is off though.')
    parser.add_argument('--ap_data_file', default='results/ap_data.pkl', type=str,
                        help='In quantitative mode, the file to save detections before calculating mAP.')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='If display not set, this resumes mAP calculations from the ap_data_file.')
    parser.add_argument('--max_images', default=-1, type=int,
                        help='The maximum number of images from the dataset to consider. Use -1 for all.')
    parser.add_argument('--output_coco_json', dest='output_coco_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this just dumps detections into the coco json file.')
    parser.add_argument('--bbox_det_file', default='results/bbox_detections.json', type=str,
                        help='The output file for coco bbox results if --coco_results is set.')
    parser.add_argument('--mask_det_file', default='results/mask_detections.json', type=str,
                        help='The output file for coco mask results if --coco_results is set.')
    parser.add_argument('--config', default='default_cfg',
                        help='The config object to use.')
    parser.add_argument('--output_web_json', dest='output_web_json', action='store_true',
                        help='If display is not set, instead of processing IoU values, this dumps detections for usage with the detections viewer web thingy.')
    parser.add_argument('--web_det_path', default='web/dets/', type=str,
                        help='If output_web_json is set, this is the path to dump detections into.')
    parser.add_argument('--no_bar', dest='no_bar', action='store_true',
                        help='Do not output the status bar. This is useful for when piping to a file.')
    parser.add_argument('--display_lincomb', default=False, type=str2bool,
                        help='If the config uses lincomb masks, output a visualization of how those masks are created.')
    parser.add_argument('--benchmark', default=False, dest='benchmark', action='store_true',
                        help='Equivalent to running display mode but without displaying an image.')
    parser.add_argument('--no_sort', default=False, dest='no_sort', action='store_true',
                        help='Do not sort images by hashed image ID.')
    parser.add_argument('--seed', default=None, type=int,
                        help='The seed to pass into random.seed. Note: this is only really for the shuffle and does not (I think) affect cuda stuff.')
    parser.add_argument('--mask_proto_debug', default=False, dest='mask_proto_debug', action='store_true',
                        help='Outputs stuff for scripts/compute_mask.py.')
    parser.add_argument('--no_crop', default=False, dest='crop', action='store_false',
                        help='Do not crop output masks with the predicted bounding box.')
    parser.add_argument('--image', default=None, type=str,
                        help='A path to an image to use for display.')
    parser.add_argument('--images', default=None, type=str,
                        help='An input folder of images and output folder to save detected images. Should be in the format input->output.')
    parser.add_argument('--video', default=None, type=str,
                        help='A path to a video to evaluate on. Passing in a number will use that index webcam.')
    parser.add_argument('--video_multiframe', default=1, type=int,
                        help='The number of frames to evaluate in parallel to make videos play at higher fps.')
    parser.add_argument('--score_threshold', default=0, type=float,
                        help='Detections with a score under this threshold will not be considered. This currently only works in display mode.')
    parser.add_argument('--dataset', default=None, type=str,
                        help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
    parser.add_argument('--detect', default=False, dest='detect', action='store_true',
                        help='Don\'t evauluate the mask branch at all and only do object detection. This only works for --display and --benchmark.')
    parser.add_argument('--display_fps', default=False, dest='display_fps', action='store_true',
                        help='When displaying / saving video, draw the FPS on the frame')
    parser.add_argument('--emulate_playback', default=False, dest='emulate_playback', action='store_true',
                        help='When saving a video, emulate the framerate that you\'d get running in real-time mode.')

    parser.set_defaults(no_bar=False, display=False, resume=False, output_coco_json=False, output_web_json=False,
                        shuffle=False,
                        benchmark=False, no_sort=False, no_hash=False, mask_proto_debug=False, crop=True, detect=False,
                        display_fps=False,
                        emulate_playback=False)

    global args
    args = parser.parse_args(argv)

    if args.output_web_json:
        args.output_coco_json = True

    if args.seed is not None:
        random.seed(args.seed)


def evaluate(net: model, dataset, train_mode=False):
    frame_times = MovingAverage()
    dataset_size = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    progress_bar = ProgressBar(30, dataset_size)

    print()

    # if not args.display and not args.benchmark:
    #     # For each class and iou, stores tuples (score, isPositive)
    #     # Index ap_data[type][iouIdx][classIdx]
    #     ap_data = {
    #         'box' : [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds],
    #         'mask': [[APDataObject() for _ in cfg.dataset.class_names] for _ in iou_thresholds]
    #     }
    #     detections = Detections()
    # else:
    #     timer.disable('Load Data')

    dataset_indices = list(range(len(dataset)))

    if args.shuffle:
        random.shuffle(dataset_indices)
    # elif not args.no_sort:
    #     # Do a deterministic shuffle based on the image ids
    #     #
    #     # I do this because on python 3.5 dictionary key order is *random*, while in 3.6 it's
    #     # the order of insertion. That means on python 3.6, the images come in the order they are in
    #     # in the annotations file. For some reason, the first images in the annotations file are
    #     # the hardest. To combat this, I use a hard-coded hash function based on the image ids
    #     # to shuffle the indices we use. That way, no matter what python version or how pycocotools
    #     # handles the data, we get the same result every time.
    #     hashed = [badhash(x) for x in dataset.ids]
    #     dataset_indices.sort(key=lambda x: hashed[x])

    dataset_indices = dataset_indices[:dataset_size]
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    cum_I, cum_U = 0, 0
    seg_total = 0.
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    print(len(dataset_indices))

    temp_Anns = np.load(cfg.valid_info, allow_pickle=True).item()['Anns']
    t_prediction = 0.0
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(), torch.ones(1, 20).cuda().long())
    try:
        pbar = tqdm(total=len(dataset_indices))
        # Main eval loop
        for it, image_idx in enumerate(dataset_indices):
            timer.reset()

            with timer.env('Load Data'):
                img, text_batch, gt, gt_masks, h, w, num_crowd = dataset.pull_item(image_idx)

                # Test flag, do not upvote
                # if cfg.mask_proto_debug:
                #     with open('scripts/info.txt', 'w') as f:
                #         f.write(str(dataset.ids[image_idx]))
                #     np.save('scripts/gt.npy', gt_masks)
                short_cut_img = img
                batch = Variable(img.unsqueeze(0))
                if args.cuda:
                    batch = batch.cuda()

            with timer.env('Network Extra'):
                start_time = time.time()
                # flag = np.where(text_batch)[0].size -2>=1 and np.where(text_batch)[0].size-2<=5
                # # flag = np.where(text_batch)[0].size -2==3
                # if not flag:
                #     continue
                preds = net(batch, torch.FloatTensor(text_batch).long().unsqueeze(0).cuda())
                diff_time = time.time() - start_time
                t_prediction += diff_time
                pbar.update(1)           
            # cv2.imwrite(root, masks.cpu().numpy() * 255)




                
                # print('Detection took {}s per image'.format(diff_time))

            # Perform the meat of the operation here depending on our mode.
            # if args.display:
            #     img_numpy = prep_display(preds, img, h, w)
            # elif args.benchmark:
            #     prep_benchmark(preds, h, w)
            # else:
                # mask_iou_cache, bbox_iou_cache, pre_mask = \
                #     prep_metrics(ap_data, preds, img, gt, gt_masks, h, w, num_crowd, dataset.Anns[image_idx],
                #                  detections)
                # shape = [h, w]
                # ratio = float(cfg.max_size) / max(shape)  # ratio  = old / new
                # new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
                # dw = (cfg.max_size - new_shape[0]) / 2  # width padding
                # dh = (cfg.max_size - new_shape[1]) / 2  # height padding
                # top, bottom = round(dh - 0.1), round(dh + 0.1)
                # left, right = round(dw - 0.1), round(dw + 0.1)
                # masks = preds[:, :, top:cfg.max_size - bottom, left:cfg.max_size - right]
                # masks = F.interpolate(masks, (h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                #masks = F.interpolate(preds, (h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

                # Binarize the masks
            preds.gt_(1e-9)
            masks = F.interpolate(preds, (h, w), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

            temp_dict = temp_Anns[image_idx]
            file_name = temp_dict['file']
            sentences = temp_dict['sent_batch'][0]
            root = "map/" + file_name + '(' + sentences + ')' + ".png"
            # print(root)
            # pic = plt.subplot(1,1,1)
            # plt.imshow(preds.data.cpu().numpy()[0,0,:,:], cmap='jet')
            # plt.show()
            # pass
            cv2.imwrite(root, masks.cpu().numpy() * 255)

            I, U = compute_mask_IU(gt_masks, masks.cpu().numpy())
            # I = mask_iou_cache[1].numpy()
            # U = mask_iou_cache[2].numpy()
            cum_I += I/U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (I / U >= eval_seg_iou)
            seg_total += 1

            # First couple of images take longer because we're constructing the graph.
            # Since that's technically initialization, don't include those in the FPS calculations.
            if it > 1:
                frame_times.add(timer.total_time())

            if args.display:
                if it > 1:
                    print('Avg FPS: %.4f' % (1 / frame_times.get_avg()))
                plt.imshow(img_numpy)
                plt.title(str(dataset.ids[image_idx]))
                plt.show()
            elif not args.no_bar:
                if it > 1: 
                    fps = 1 / frame_times.get_avg()
                else:
                    fps = 0
                progress = (it + 1) / dataset_size * 100
                progress_bar.set_val(it + 1)
                print('\rProcessing Images  %s %6d / %6d (%5.2f%%)    %5.2f fps        '
                      % (repr(progress_bar), it + 1, dataset_size, progress, fps), end='')

        pbar.close()
        # Print results
        print('\nSegmentation evaluation (without DenseCRF):')
        result_str = ''
        for n_eval_iou in range(len(eval_seg_iou_list)):
            result_str += 'precision@%s = %f\n' % \
                          (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] / seg_total)
        result_str += 'mean IoU = %f\n' % (cum_I / seg_total)
        print(result_str)

        print("Prediction time: {}. Average {}/image {}FPS".format(
            t_prediction, t_prediction / len(dataset_indices), 1.0/(t_prediction / len(dataset_indices))))

        # if not args.display and not args.benchmark:
        #     print()
        #     if args.output_coco_json:
        #         print('Dumping detections...')
        #         if args.output_web_json:
        #             detections.dump_web()
        #         else:
        #             detections.dump()
        #     else:
        #         if not train_mode:
        #             print('Saving data...')
        #             with open(args.ap_data_file, 'wb') as f:
        #                 pickle.dump(ap_data, f)
        #
        #         return calc_map(ap_data)
        # elif args.benchmark:
        #     print()
        #     print()
        #     print('Stats for the last frame:')
        #     timer.print_stats()
        #     avg_seconds = frame_times.get_avg()
        #     print('Average: %5.2f fps, %5.2f ms' % (1 / frame_times.get_avg(), 1000*avg_seconds))

    except KeyboardInterrupt:
        print('Stopping...')
    return result_str, cum_I / cum_U


if __name__ == '__main__':
    parse_args()

    if args.config is not None:
        set_cfg(args.config)

    if args.trained_model == 'interrupt':
        args.trained_model = SavePath.get_interrupt('weights/')
    elif args.trained_model == 'latest':
        args.trained_model = SavePath.get_latest('weights/', cfg.name)

    if args.config is None:
        model_path = SavePath.from_str(args.trained_model)
        # TODO: Bad practice? Probably want to do a name lookup instead.
        args.config = model_path.model_name + '_config'
        print('Config not specified. Parsed %s from the file name.\n' % args.config)
        set_cfg(args.config)
    args.trained_model = "/home/ubuntu/Desktop/temp/unc_full_nonlocal_dynamic_24_125000.pth"
    if args.detect:
        cfg.eval_mask_branch = False

    if args.dataset is not None:
        set_dataset(args.dataset)

    with torch.no_grad():
        if args.cuda:
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if args.image is None and args.video is None and args.images is None:
            dataset = ReferDataset(image_path=cfg.valid_images,
                                   info_file=cfg.valid_info,
                                   img_size=cfg.max_size, resize_gt=False)
            # prep_coco_cats()
        else:
            dataset = None

        print('Loading model...', end='')
        net = model()
        net.load_weights(args.trained_model)
        net.eval()
        print(' Done.')

        if args.cuda:
            net = net.cuda()

        evaluate(net, dataset)
