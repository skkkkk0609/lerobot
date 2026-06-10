"""Quick test to isolate camera connection issue."""
import os, platform, time, cv2

if platform.system() == "Windows" and "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ:
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

for idx in [1, 2]:
    print(f"\n=== Testing camera index {idx} ===")
    
    # Test 1: CAP_MSMF with MJPG
    print("Test 1: CAP_MSMF + MJPG + FPS first")
    t0 = time.time()
    cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
    print(f"  Open: {time.time()-t0:.1f}s, opened={cap.isOpened()}")
    if cap.isOpened():
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        print(f"  Set MJPG: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"  Set FPS: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        print(f"  Set Width: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"  Set Height: {time.time()-t0:.1f}s")
        t0 = time.time()
        ret, frame = cap.read()
        print(f"  Read frame: {time.time()-t0:.1f}s, success={ret}, shape={frame.shape if ret else 'N/A'}")
        cap.release()
    
    # Test 2: CAP_DSHOW (no MJPG)
    print("Test 2: CAP_DSHOW, no MJPG")
    t0 = time.time()
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    print(f"  Open: {time.time()-t0:.1f}s, opened={cap.isOpened()}")
    if cap.isOpened():
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"  Set FPS: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        print(f"  Set Width: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"  Set Height: {time.time()-t0:.1f}s")
        t0 = time.time()
        ret, frame = cap.read()
        print(f"  Read frame: {time.time()-t0:.1f}s, success={ret}, shape={frame.shape if ret else 'N/A'}")
        cap.release()
    
    # Test 3: CAP_ANY (default)
    print("Test 3: CAP_ANY (default)")
    t0 = time.time()
    cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
    print(f"  Open: {time.time()-t0:.1f}s, opened={cap.isOpened()}")
    if cap.isOpened():
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        print(f"  Set MJPG: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"  Set FPS: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        print(f"  Set Width: {time.time()-t0:.1f}s")
        t0 = time.time()
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"  Set Height: {time.time()-t0:.1f}s")
        t0 = time.time()
        ret, frame = cap.read()
        print(f"  Read frame: {time.time()-t0:.1f}s, success={ret}, shape={frame.shape if ret else 'N/A'}")
        cap.release()

print("\n=== Done ===")
