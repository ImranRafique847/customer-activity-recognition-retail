"""
models/register.py
==================
Registers custom modules (CBAM etc.) with ultralytics so they can be
referenced by name in YAML model configs.

Call register_custom_modules() before loading any YOLO model.
"""

from models.cbam import CBAM, ChannelAttention, SpatialAttention


def register_custom_modules():
    """
    Inject custom modules into ultralytics module namespace.
    Must be called before YOLO('yolo11s-cbam.yaml') or model.train().
    """
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules.block as block

    # Register in tasks module (used by YOLO parser)
    tasks.CBAM = CBAM
    tasks.ChannelAttention = ChannelAttention
    tasks.SpatialAttention = SpatialAttention

    # Register in block module (used by some YAML parsers)
    block.CBAM = CBAM
    block.ChannelAttention = ChannelAttention
    block.SpatialAttention = SpatialAttention

    # Register in ultralytics __init__ namespace
    import ultralytics.nn.modules as modules
    modules.CBAM = CBAM
    modules.ChannelAttention = ChannelAttention
    modules.SpatialAttention = SpatialAttention

    print("Custom modules registered: CBAM, ChannelAttention, SpatialAttention")
