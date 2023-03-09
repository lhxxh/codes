import os

strs = ('pip install mmcv==0.6.2')
os.system(strs)

strs = ('conda install -c conda-forge terminaltables')
os.system(strs)

strs = ('cd models/py_utils/_cpools && python setup.py install --user')
os.system(strs)

strs = ('cd models/py_utils/roi_extractors/ops/roi_align && python setup.py install --user')
os.system(strs)

strs = ('cd data/coco/PythonAPI && make')
os.system(strs)

strs = ('cd external && make')
os.system(strs)

strs = ('cd models/py_utils/bbox/nms && python setup.py install --user')
os.system(strs)

strs = ('conda install -c anaconda numpy')
os.system(strs)

