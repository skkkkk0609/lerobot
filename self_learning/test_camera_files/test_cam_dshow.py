"""Test DSHOW with full config."""
import os, platform, cv2, time

if platform.system() == "Windows":
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

for idx in [1, 2]:
    print(f"\nCamera {idx} with DSHOW:")
    t0 = time.time()
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    print(f"  Open: {time.time()-t0:.1f}s, ok={cap.isOpened()}")
    if not cap.isOpened():
        continue
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FPS, 30)
    print(f"  Set FPS=30: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    print(f"  Set W=640: {time.time()-t0:.1f}s")
    
    t0 = time.time()
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"  Set H=480: {time.time()-t0:.1f}s")
    
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"  Actual: fps={actual_fps}, {actual_w}x{actual_h}")
    
    # Read 5 frames to test streaming
    t0 = time.time()
    for i in range(5):
        ret, frame = cap.read()
    dt = time.time() - t0
    print(f"  5 frames in {dt:.2f}s ({5/dt:.1f} fps), shape={frame.shape if ret else 'N/A'}")
    cap.release()

print("\nDone")
