import argparse
from Face_detector.utils import *
from Face_detector.yolo.yolo import YOLO
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from PIL import Image

# Give the configuration and weight files for the model and load the network
# using them.


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='./Face_detector/model-weights/YOLO_Face.h5',
                        help='path to model weights file')
    parser.add_argument('--anchors', type=str, default='./Face_detector/cfg/yolo_anchors.txt',
                        help='path to anchor definitions')
    parser.add_argument('--classes', type=str, default='./Face_detector/cfg/face_classes.txt',
                        help='path to class definitions')
    parser.add_argument('--score', type=float, default=0.5,
                        help='the score threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='the iou threshold')
    parser.add_argument('--img-size', type=list, action='store',
                        default=(416, 416), help='input image size')
    args = parser.parse_args()
    return args


def prepare_net(mode):
    if mode == "cpu":
        net = cv2.dnn.readNetFromDarknet('./Face_detector/cfg/yolov3-face.cfg',
                                         "./Face_detector/model-weights/yolov3-wider_16000.weights")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    if mode == "gpu":
        args = get_args()
        net = YOLO(args)
        return net


def detect_face(frame, net, mode):
    if mode == "cpu":
        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                     [0, 0, 0], 1, crop=False)

        net.setInput(blob)

        outs = net.forward(get_outputs_names(net))

        faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
        return faces, frame
    if mode == "gpu":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        res_image, boxes = net.detect_image(frame)

        res_image = np.asarray(res_image)
        res_image = cv2.cvtColor(res_image, cv2.COLOR_RGB2BGR)
        return boxes, res_image


def draw_faces(frame, faces, cls):
    if cls == 0:
        fas = "spoof"
    elif cls == 1:
        fas = "real"
    if cls is not None:
        info = [
            ('number of faces detected', '{}'.format(len(faces))),
            ("Face is ", "{}".format(fas))
        ]
    else:
        info = [
            ('number of faces detected', '{}'.format(len(faces))),
        ]
    for (i, (txt, val)) in enumerate(info):
        text = '{}: {}'.format(txt, val)
        cv2.putText(frame, text, (50, (i * 90) + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_RED, 4)
    return frame


