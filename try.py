import pynvml

pynvml.nvmlInit()

handle = pynvml.nvmlDeviceGetHandleByIndex(1)  # 第0块显卡
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(meminfo.used)  # 查看显卡显存使用量

import gc
import objgraph

gc.collect()  # 变量回收
objgraph.show_growth()  # 显示当前变量类型引用次数 及增加引用次数
objgraph.show_most_common_types(limit=50)  # 显示引用次数最多的50个变量类型