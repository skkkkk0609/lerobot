"""Python 3.13 + Windows 的 pathlib/importlib 性能补丁

必须在所有其他 import 之前执行。修复 PackagePath.__str__ 的
AttributeError 导致 importlib.metadata 全部操作都超慢的问题。
"""
import pathlib._local
import importlib.metadata

# ---- 修复 1: pathlib PackagePath 的 _str bug ----
_orig_path_str = pathlib._local.PureWindowsPath.__str__


def _safe_str(self):
    try:
        return _orig_path_str(self)
    except AttributeError:
        return "\\"  # 返回一个安全的默认值


pathlib._local.PureWindowsPath.__str__ = _safe_str

# ---- 修复 2: 预热 packages_distributions ----
importlib.metadata.packages_distributions()
