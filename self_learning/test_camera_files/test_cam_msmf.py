"""Test cameras with MSMF + HW_TRANSFORMS fix."""
import os, platform, cv2, time

if platform.system() == "Windows":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

for idx in [1, 2]:
    print(f"\nCamera {idx} MSMF:")
    t0 = time.time()
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    dt = time.time() - t0
    ok = cap.isOpened()
    print(f"  Open: {dt:.1f}s, ok={ok}")
    if not ok:
        cap.release()
        continue
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    print(f"  Set MJPG: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"  Set FPS=30: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    print(f"  Set W=640: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  Set H=480: {time.time()-t0:.1f}s")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    print(f"  Actual: fps={fps}, {w}x{h}, fourcc={fourcc_str}")
    
    t0 = time.time()
    for i in range(10):
        ret, frame = cap.read()
    dt = time.time() - t0
    print(f"  10 frames in {dt:.2f}s ({10/dt:.1f} fps)")
    cap.release()

print("\nDone")
