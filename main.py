import os
import sys
import cv2
import platform
import argparse
import numpy as np

from openvino.runtime import Core

MODEL_BASE_PATH: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")

DETECT_MODEL_PATH: str  = os.path.join(MODEL_BASE_PATH, "detect_model.xml")
RECOG_MODEL_PATH_1: str = os.path.join(MODEL_BASE_PATH, "recog_model_1.xml")
RECOG_MODEL_PATH_2: str = os.path.join(MODEL_BASE_PATH, "recog_model_2.xml")
RECOG_MODEL_PATH_3: str = os.path.join(MODEL_BASE_PATH, "recog_model_3.xml")

IMAGE_PATH: str = os.path.join(os.path.abspath(os.path.dirname(__file__)), "images")

ID: int         = 0
CAM_WIDTH: int  = 640
CAM_HEIGHT: int = 360 
FPS: int        = 30


def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")


def preprocess(image: np.ndarray, width: int, height: int, model_name: str="arcface") -> np.ndarray:
    if model_name == "arcface" or model_name == "sphereface":
        image = cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
    else:
        image = cv2.resize(src=image, dsize=(width, height), interpolation=cv2.INTER_AREA)
    return np.expand_dims(image, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b.reshape(-1, 1)) / (np.linalg.norm(a) * np.linalg.norm(b))


def setup(target: str, model_path: str) -> tuple:
    ie = Core()
    model = ie.read_model(model=model_path)
    model = ie.compile_model(model=model, device_name=target)

    input_layer = next(iter(model.inputs))
    output_layer = next(iter(model.outputs))

    return model, input_layer, output_layer, \
           (input_layer.shape[0], input_layer.shape[1], input_layer.shape[2], input_layer.shape[3])


def detect_faces(
    model, 
    output_layer, 
    image: np.ndarray, 
    w: int, 
    h: int) -> tuple:

    result = model(inputs=[image])[output_layer].squeeze()

    label_indexes: list = []
    probs: list = []
    boxes: list = []

    if result[0][0] == -1:
        return 0, None, None    
    else:
        for i in range(result.shape[0]):
            if result[i][0] == -1:
                break
            else:
                label_indexes.append(int(result[i][1]))
                probs.append(result[i][2])
                boxes.append([int(result[i][3] * w), \
                              int(result[i][4] * h), \
                              int(result[i][5] * w), \
                              int(result[i][6] * h)])
        
    return label_indexes, probs, boxes


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", type=str, default="Test_1.jpg", help="Image Filename")
    parser.add_argument("--target", "-t", type=str, default="CPU", help="Target Device for Inference")
    parser.add_argument("--model", "-m", type=str, default="arcface", help="Model to Use (arcface, facenet, sphereface)")
    args = parser.parse_args()

    assert args.filename in os.listdir(IMAGE_PATH), "File not Found"
    assert args.target in ["CPU", "GPU"], "Invalid Target Device"

    image = cv2.imread(os.path.join(IMAGE_PATH, args.filename), cv2.IMREAD_COLOR)
    temp_image = image.copy()
    h, w, _ = image.shape

    d_model, _, d_output_layer, (_, _, d_H, d_W) = setup(args.target, DETECT_MODEL_PATH)
    
    if args.model == "arcface":
        r_model, _, r_output_layer, (_, _, r_H, r_W) = setup(args.target, RECOG_MODEL_PATH_1)
    elif args.model == "facenet":
        r_model, _, r_output_layer, (_, r_H, r_W, _) = setup(args.target, RECOG_MODEL_PATH_2)
    elif args.model == "sphereface":
        r_model, _, r_output_layer, (_, _, r_H, r_W) = setup(args.target, RECOG_MODEL_PATH_3)

    image = preprocess(image, d_W, d_H)
    _, _, boxes = detect_faces(d_model, d_output_layer, image, w, h)

    face_image = preprocess(temp_image[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2], :], r_W, r_H, args.model)
    reference_embeddings = r_model(inputs=[face_image])[r_output_layer]

    if platform.system() != "Windows":
        cap = cv2.VideoCapture(ID)
    else:
        cap = cv2.VideoCapture(ID, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    while True:
        ret, frame = cap.read()
        disp_frame = frame.copy()
        if not ret: break
        
        frame = preprocess(frame, d_W, d_H)
        _, _, boxes = detect_faces(d_model, d_output_layer, frame, CAM_WIDTH, CAM_HEIGHT)
        face_frame = disp_frame[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2], :]

        if face_frame.shape[0] < 32 or face_frame.shape[1] < 32:
            # print("ROI to small to detect")
            cv2.putText(disp_frame, "ROI to small to detect", org=(25, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(0, 0, 255))
        else:
            face_frame = preprocess(face_frame, r_W, r_H, args.model)
            embeddings = r_model(inputs=[face_frame])[r_output_layer]

            cs = cosine_similarity(reference_embeddings, embeddings)[0][0]

            # print(f"{cs:.2f}")
            cv2.putText(disp_frame, f"{cs:.2f}", org=(25, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=1, color=(0, 255, 0))
        
        cv2.imshow("Feed", disp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main() or 0)
