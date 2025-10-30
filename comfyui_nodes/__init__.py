"""
Jimeng API ComfyUI Custom Nodes
即梦API ComfyUI自定义节点包
"""

from .jimeng_unified import JimengUnified

# 节点映射
NODE_CLASS_MAPPINGS = {
    "Jimeng_Unified": JimengUnified,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Unified": "Jimeng (即梦-智能生图)",
}

# 节点版本
__version__ = "2.0.0"

# 导出所有
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

