"""读写层：视频拆帧、产物落盘、JSON / npz / GLB 导出。

注意：任务清单里写作 ``io/``，但 Python 标准库已有 ``io`` 模块，
同名包会覆盖标准库导致 torch / logging 等依赖直接崩溃，
因此这里改用 ``dataio``。
"""
