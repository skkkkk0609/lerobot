"""Test MSMF backend properly - env var must be set BEFORE cv2 import."""
import platform, os
if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2, time

for idx in [1, 2]:
    print(f"\nCamera {idx} MSMF:")
    t0 = time.time()
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    ok = cap.isOpened()
    print(f"  Open: {time.time()-t0:.1f}s, ok={ok}")
    if not ok:
        cap.release()
        continue
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"  MJPG: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"  FPS: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    print(f"  W: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  H: {time.time()-t0:.1f}s")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fc_str = "".join([chr((fc>>8*i)&0xFF) for i in range(4)])
    print(f"  Result: {fps}fps, {int(w)}x{int(h)}, {fc_str}")
    
    t0 = time.time()
    for _ in range(10):
        cap.read()
    print(f"  10 reads: {time.time()-t0:.2f}s ({10/max(time.time()-t0,0.01):.0f}fps)")
    cap.release()

print("Done")
