import cv2
import sys
import time
from pathlib import Path
import shutil
import os
import torch
from collections import OrderedDict
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import pickle

from opts import arg_parser
from models import build_model
from utils.video_transforms import *
from utils.video_dataset import VideoDataSet
from utils.utils import build_dataflow, AverageMeter


def inference():
    data_path = 'data_inference'
    videos_path = str(Path(data_path).joinpath('videos'))
    txts_path = str(Path(data_path).joinpath('txts'))
    video_name = 'laugh2'

    # variables for creating result video
    size = 600
    max_image_counter = 48
    video_counter = 0
    image_counter = 0
    max_frames_show_predictions = 24
    current_frames_show_predictions = 0

    num_classes = 7
    image_tmpl = '{:05d}.jpg'
    filename_seperator = ' '
    filter_video = 0

    # remove data and recreate folders
    shutil.rmtree(videos_path)
    shutil.rmtree(txts_path)
    Path(videos_path).mkdir(parents=True, exist_ok=True)
    Path(txts_path).mkdir(parents=True, exist_ok=True)

    args = get_args(num_classes=num_classes)
    model, augmentor = create_model(args)

    cap = cv2.VideoCapture(str(Path(data_path).joinpath(video_name + '.avi')))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print('frames_count', length)

    if not cap.isOpened():
        print("Error opening the video file. Please double check your "
              "file path for typos. Or move the movie file to the same location as this script/notebook")
        sys.exit()

    class_name = ''
    prev_frame_time = 0

    new_frame_time = 0

    img_array = []
    size_shape = ()

    while cap.isOpened():
        # Read the video file.
        ret, image = cap.read()

        # If we got frames, show them.
        if ret == True:
            time.sleep(1 / fps)

            if video_counter == 0:
                folder_path = Path(videos_path).joinpath(str(video_counter) + '_video_' + video_name)
                Path(folder_path).mkdir(parents=True, exist_ok=True)

            image_counter += 1
            image_name = f"{image_counter:05d}.jpg"
            image_path = Path(videos_path).joinpath(str(video_counter) + '_video_' + video_name).joinpath(
                image_name)
            cv2.imwrite(str(image_path), image)

            if image_counter == max_image_counter:
                start = time.time()
                current_frames_show_predictions = 0

                txt_file_path = Path(txts_path).joinpath(str(video_counter) + '_test_' + video_name + '.txt')
                with open(txt_file_path, "w") as text_file:
                    rec = str(Path(videos_path).joinpath(str(video_counter) +
                                                         '_video_' + video_name)) + ' 1 ' + str(max_image_counter)
                    print(rec, file=text_file)

                # prediction
                class_name = get_prediction(args,
                                            test_list_name=str(Path('data_inference/txts').joinpath(str(video_counter) +
                                                                                                    '_test_' + video_name + '.txt')),
                                            image_tmpl=image_tmpl,
                                            filename_seperator=filename_seperator,
                                            filter_video=filter_video,
                                            model=model,
                                            augmentor=augmentor)

                image_counter = 0
                video_counter += 1
                folder_path = Path(videos_path).joinpath(str(video_counter) + '_video_' + video_name)
                Path(folder_path).mkdir(parents=True, exist_ok=True)

                end = time.time()

                print('Prediction time: {}'.format(end - start))

            if class_name != '' and current_frames_show_predictions < max_frames_show_predictions:
                current_frames_show_predictions += 1
                image = cv2.putText(image, 'Detected ' + class_name, (int(image.shape[0] * 0.1), 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 0, 255), 1, cv2.LINE_AA)

            image = resize_with_aspect_ratio(image, width=size)

            # show fps
            # new_frame_time = time.time()
            # fpss = int(1 / (new_frame_time - prev_frame_time))
            # prev_frame_time = new_frame_time
            # cv2.putText(image, str(fpss), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

            (H, W) = image.shape[:2]
            size_shape = (W, H)

            cv2.imshow('image', image)

            img_array.append(image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    print("Saving video...")
    out = cv2.VideoWriter(str(Path(data_path).joinpath('result.avi')), cv2.VideoWriter_fourcc(*'DIVX'), 18, size_shape)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()


def get_args(num_classes):
    parser = arg_parser()
    args = parser.parse_args()

    args.datadir = 'data_inference'

    args.num_classes = num_classes
    args.batch_size = 1
    args.backbone_net = 'resnet'
    args.modality = 'rgb'
    args.dataset = 'test_dataset'
    args.pretrained = 'test_dataset-rgb-resnet-18-ts-max-f16-cosine-bs2-e100/model_best.pth.tar'

    return args


def create_model(args):
    cudnn.benchmark = True

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args, test_mode=True)
    mean = model.mean(args.modality)
    std = model.std(args.modality)

    # overwrite mean and std if they are presented in command
    if args.mean is not None:
        if args.modality == 'rgb':
            if len(args.mean) != 3:
                raise ValueError("When training with rgb, dim of mean must be three.")
        elif args.modality == 'flow':
            if len(args.mean) != 1:
                raise ValueError("When training with flow, dim of mean must be one.")
        mean = args.mean

    if args.std is not None:
        if args.modality == 'rgb':
            if len(args.std) != 3:
                raise ValueError("When training with rgb, dim of std must be three.")
        elif args.modality == 'flow':
            if len(args.std) != 1:
                raise ValueError("When training with flow, dim of std must be one.")
        std = args.std

    model = model.cuda()
    model.eval()

    if args.pretrained is not None:

        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        # delete module word
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v

        checkpoint['state_dict'] = new_state_dict

        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(args.pretrained)
    else:
        print("=> creating model '{}'".format(arch_name))

    model = torch.nn.DataParallel(model).cuda()

    augments = []

    augments += [
        GroupScale(args.input_size),
        GroupCenterCrop(args.input_size)
    ]

    augments += [
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]

    augmentor = transforms.Compose(augments)

    return model, augmentor


def get_prediction(args,
                   test_list_name,
                   image_tmpl,
                   filename_seperator,
                   filter_video,
                   model,
                   augmentor):
    data_list = test_list_name
    val_dataset = VideoDataSet(args.datadir,
                               data_list,
                               args.groups,
                               args.frames_per_group,
                               num_clips=args.num_clips,
                               modality=args.modality,
                               image_tmpl=image_tmpl,
                               dense_sampling=args.dense_sampling,
                               fixed_offset=not args.random_sampling,
                               transform=augmentor,
                               is_train=False,
                               test_mode=not args.evaluate,
                               seperator=filename_seperator,
                               filter_video=filter_video)

    data_loader = build_dataflow(val_dataset,
                                 is_train=False,
                                 batch_size=args.batch_size,
                                 workers=args.workers)

    # switch to evaluate mode
    model.eval()

    for i, (video, label) in enumerate(data_loader):
        output = eval_a_batch(video, model)

        output = output.data.cpu().numpy().copy()
        # print('output', output)
        predictions = np.argsort(output, axis=1)
        for ii in range(len(predictions)):
            temp = predictions[ii][::-1][:5]
            preds = [str(pred) for pred in temp]

            label_encoder = pickle.load(open("dataset_dir/label_encoder.pkl", 'rb'))
            actual_label = label_encoder.classes_[int(preds[0])]

    return actual_label


def eval_a_batch(data, model):
    with torch.no_grad():
        batch_size = data.shape[0]

        data = data.view((batch_size, -1) + data.size()[2:])
        result = model(data)

    return result


def resize_with_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)


if __name__ == '__main__':
    inference()
