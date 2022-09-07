### **OpenVINO Facial Recognition Application**

- Place reference images in `images` folder
- Download [models](https://drive.google.com/drive/folders/1nq-C_pSKA309OaeiaUmv_-9cUyHYyD02?usp=sharing) and place it in the `models` folder
- Models used 
    - `face-detection-retail-0044`
    - recog_model_1 (arcface) = `face-recognition-resnet100-arcface-onnx`
    - recog_model_2 (facenet) = `facenet-20180408-102900`
    - recog_model_3 (sphereface) = `Sphereface`
- CLI Arguments
    - `--filename | -f` - Filename of the Reference Image (placed in the `images` folder)
    - `--target | -t`   - Target device to perform inference on (Available: `CPU` and `GPU`, Default: `CPU`)
    - `--model | -m`    - Recognition Model to Use (Available: `arcface`, `facenet` and `sphereface`, Default: `facenet`)