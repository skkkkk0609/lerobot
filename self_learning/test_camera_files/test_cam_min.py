"""Minimal test - try each backend.""" 
import os, platform, cv2

if platform.system() == "Windows":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

for idx in [1, 2]:
    for name, backend in [("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]:
        print(f"\nCamera {idx} {name}: ", end="", flush=True)
        cap = cv2.VideoCapture(idx, backend)
        ok = cap.isOpened()
        cap.release()
        print(f"open={'OK' if ok else 'FAIL'}")
print("Done")
