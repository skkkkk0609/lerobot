import sys
import os

print("=" * 50)
print("环境测试脚本")
print("=" * 50)
print()

# 1. 检查 Python 版本
print(f"Python 版本: {sys.version}")
print()

# 2. 检查 SDK 路径
sdk_path = os.environ.get(
    "GENKIARM_SDK_PATH",
    r"D:\arm_robot_begin\机械臂资料00\FTServo_Python-main\FTServo_Python-main"
)
print(f"SDK 路径: {sdk_path}")
print(f"路径是否存在: {os.path.exists(sdk_path)}")

if os.path.exists(sdk_path):
    print(f"路径内容: {os.listdir(sdk_path)}")
print()

# 3. 尝试导入 SDK
try:
    sys.path.append(sdk_path)
    from scservo_sdk import *
    print("✅ SDK 导入成功")
    print(f"  COMM_SUCCESS = {COMM_SUCCESS}")
except Exception as e:
    print(f"❌ SDK 导入失败: {e}")
    import traceback
    traceback.print_exc()
print()

# 4. 检查 websockets 库
try:
    import websockets
    print("✅ websockets 库已安装")
except ImportError:
    print("❌ websockets 库未安装，请运行: pip install websockets")
print()

print("测试完成！")
