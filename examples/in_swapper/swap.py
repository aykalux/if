import argparse
import argparse
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from .swapper import get as get_swapper


def main(args):
    app = FaceAnalysis(name='buffalo_l')
    app.prepare(ctx_id=0, det_size=(640, 640))
    swapper = get_swapper(args.model)

    img = cv2.imread(args.target)
    faces = app.get(img)
    if len(faces) < 2:
        raise ValueError("Need at least two faces to swap")
    source_face = faces[0]
    res = img.copy()
    for face in faces[1:]:
        res = swapper.get(res, face, source_face, paste_back=True)
    cv2.imwrite(args.output, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--output', type=str, default='swap_out.jpg')
    parser.add_argument('--model', type=str, default='inswapper_1024.onnx')
    args = parser.parse_args()
    main(args)
