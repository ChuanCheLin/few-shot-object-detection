import cv2
from tqdm import tqdm

import argparse
import glob
import multiprocessing as mp
import os
import time

from predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from fsdet.config import get_cfg

name2id = { 1:'brownblight', 2:'algal', 3: 'blister', 4: 'sunburn', 5: 'fungi_early', 6: 'roller',
            7: 'moth', 8: 'tortrix', 9: 'flushworm', 10: 'caloptilia', 11: 'mosquito_early', 12: 'mosquito_late',
            13: 'miner', 14: 'thrips', 15: 'tetrany', 16: 'formosa', 17: 'other'}


def write_xml(w, h, coordinates, classid, jpgname, xml_name, xml_dir):
    import xml.dom.minidom
    doc = xml.dom.minidom.Document() 
    annotation = doc.createElement('annotation') 
    doc.appendChild(annotation)
    # folder
    folder = doc.createElement('folder')
    annotation.appendChild(folder)
    folder.appendChild(doc.createTextNode('auto'))
    # filename
    filename = doc.createElement('filename')
    filename.appendChild(doc.createTextNode(jpgname))
    annotation.appendChild(filename)
    # path
    path = doc.createElement('path')
    annotation.appendChild(path)
    path.appendChild(doc.createTextNode('auto'))
    # source
    source = doc.createElement('source')
    annotation.appendChild(source)
    database = doc.createElement('database')
    source.appendChild(database)
    database.appendChild(doc.createTextNode('Unknown'))
    # size
    nodeSize = doc.createElement('size')
    annotation.appendChild(nodeSize)
    width = doc.createElement('width')
    height = doc.createElement('height')
    depth = doc.createElement('depth')
    nodeSize.appendChild(width)
    nodeSize.appendChild(height)
    nodeSize.appendChild(depth)
    width.appendChild(doc.createTextNode(str(w)))
    height.appendChild(doc.createTextNode(str(h)))
    depth.appendChild(doc.createTextNode(str(3)))
    # segmented
    segmented = doc.createElement('segmented')
    annotation.appendChild(segmented)
    segmented.appendChild(doc.createTextNode('0'))
    # object
    for i in range(len(coordinates)) :
        nodeObject = doc.createElement('object')
        annotation.appendChild(nodeObject)
        # name
        name = doc.createElement('name')
        nodeObject.appendChild(name)
        #print((classid[i]))
        name.appendChild(doc.createTextNode(name2id.get(classid[i] + 1)))
        # pose
        pose = doc.createElement('pose')
        nodeObject.appendChild(pose)
        pose.appendChild(doc.createTextNode('Unspecified'))
        # truncated
        truncated = doc.createElement('truncated')
        nodeObject.appendChild(truncated)
        truncated.appendChild(doc.createTextNode('0'))
        # difficult
        difficult = doc.createElement('difficult')
        nodeObject.appendChild(difficult)
        difficult.appendChild(doc.createTextNode('0'))


        # bnb box
        nodebnb = doc.createElement('bndbox')
        nodeObject.appendChild(nodebnb)
        node1 = doc.createElement('xmin')
        node1.appendChild(doc.createTextNode(str(int(coordinates[i][0]))))

        node2 = doc.createElement("ymin")
        node2.appendChild(doc.createTextNode(str(int(coordinates[i][1]))))

        node3 = doc.createElement("xmax")
        node3.appendChild(doc.createTextNode(str(int(coordinates[i][2]))))

        node4 = doc.createElement("ymax")
        node4.appendChild(doc.createTextNode(str(int(coordinates[i][3]))))

        nodebnb.appendChild(node1)
        nodebnb.appendChild(node2)
        nodebnb.appendChild(node3)
        nodebnb.appendChild(node4)
    if(os.path.isdir(xml_dir)==False):
        os.mkdir(xml_dir)
    xml_path = xml_dir + xml_name
    fp = open(xml_path, 'w')
    doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="FsDet demo for builtin models"
    )
    parser.add_argument(
        "--config-file",
        default="configs/COCO-detection/faster_rcnn_R_101_FPN_ft_all_1shot.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--unlabel_jpg_dir",
        help="Path to jpg file that needs to be labeled.",
    )
    parser.add_argument(
        "--ouput_xml_dir",
        help="Path to generated xml file.",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    # unlabeled jpg dir
    jpg_dir = args.unlabel_jpg_dir
    if args.input:
        f = open(args.input[0], 'r')
        for jpg in tqdm(f.readlines()):
            # use PIL, to be consistent with evaluation
            jpg = jpg.rstrip('\n')
            image_name = jpg.rstrip('.jpg')
            xml_name = image_name + '.xml'
            path = jpg_dir + jpg
            
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            # xml
            height = (predictions["instances"].image_size[0])
            width = (predictions["instances"].image_size[1])
            # x, y
            coordinates = predictions["instances"]._fields.get('pred_boxes').tensor.tolist()
            # classes
            classes = predictions["instances"]._fields.get('pred_classes').tolist()

            write_xml(w = width, h = height, coordinates = coordinates, classid = classes, jpgname = jpg, xml_name = xml_name, xml_dir=args.ouput_xml_dir)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(
                        len(predictions["instances"])
                    )
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if not os.path.isdir(args.output):
                    os.mkdir(args.output)

                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(
                    args.output, os.path.basename(path)
                )
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(
                    WINDOW_NAME, visualized_output.get_image()[:, :, ::-1]
                )
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


