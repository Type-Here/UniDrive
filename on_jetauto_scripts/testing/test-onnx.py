# test_model.py -- salva questo sul Jetson e runnalo
import numpy as np
import cv2


# Carica una immagine reale
img = cv2.imread("frame_test.jpg")   # usa uno dei frame di training
print("Image shape:", img.shape)

# Preprocessing identico a lane_follower
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

crop_px = int(img.shape[0] * 0.45)
cropped = img[crop_px:, :]
resized = cv2.resize(cropped, (640, 256))
rgb     = resized[:, :, ::-1]   # BGR->RGB (opencv carica BGR)
norm    = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
chw     = norm.transpose(2, 0, 1)[np.newaxis]   # (1,3,256,640)

print("Input range: min=%.3f max=%.3f mean=%.3f" % (
    chw.min(), chw.max(), chw.mean()))

# Salva l'input preprocessato per debug
np.save("input_debug.npy", chw)
print("Input saved to input_debug.npy")

# Carica ONNX (piu' semplice per debug, non serve TRT)
import onnxruntime as ort
sess = ort.InferenceSession("model.onnx",
    providers=["CPUExecutionProvider"])
out  = sess.run(None, {"pixel_values": chw})[0]
print("Output shape:", out.shape)
print("Unique classes:", np.unique(out))
print("Class counts:", dict(zip(*np.unique(out, return_counts=True))))