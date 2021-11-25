import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmdet.apis import train_detector, inference_detector, init_detector, show_result_pyplot
from mmcv import Config

cfg = Config.fromfile('./configs/detectors/detectors_cascade_rcnn_r50_1x_coco.py')
cfg.load_from = './detection_model/epoch_4.pth'

img = mmcv.imread('/kaggle/working/test/0044.jpg')

model.cfg = cfg
result = inference_detector(model, img)
show_result_pyplot(model, img, result)
