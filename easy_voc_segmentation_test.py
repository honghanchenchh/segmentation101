# file name:  easy_voc_segmentation_test.py
# Author: honghan chen  陈洪瀚

from PIL import Image
import numpy as np 
import torchvision.utils as vutils
import torchvision.models as models
import torchvision.transforms as T 
import matplotlib.pyplot as plt

# 1.
# load image data
# 读图像数据, 将原图和分割标准图像路径改为数据存放路径,<user>更改为自定的存放路径
img_path = '/home/<user>/VOC/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg'
label_path = '/home/<user>/VOC/VOCdevkit/VOC2012/SegmentationClass/2007_000033.png'
img = Image.open(img_path)
label_img = Image.open(label_path)

# 2
# get input data batch 将输入图像变换为模型需要的batchg
test_transform = T.Compose(
    [T.ToTensor(),
    T.Normalize(mean = [0.485,0.456,0.406],std=[0.229,0.224,0.225])]
)
input_tensor = test_transform(img)
input_batch = input_tensor.unsqueeze(0)
# [Ci x Hi x Wi]->[Ni x Ci x Hi x Wi] 彩色图像3通道，变换到batch 
#Ni -> the batch size 
#Ci -> the number of channels (which is 3) 图像通道
#Hi -> the height of the image 高
#Wi -> the width of the image 宽

# 3
# load pretrained segmentation model and predict input data batch
# wait model downloading for the firt time.
seg_model = models.segmentation.fcn_resnet101(pretrained=True)
# 上面使用了全卷积网络 FCN, 如果姚使用DeepLabV3, 把上面注释掉，下面取消注释
#seg_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
seg_model.eval()
output = seg_model(input_batch)['out'][0]
output_prediction = output.argmax(0)

# 4
# Transform output label to VOC segmentation label format
label_prediction = output_prediction.numpy().astype(np.uint8)
img_prediction = T.ToPILImage()(label_prediction).convert('P')
# change gray image to image with color palette 
# 得到的灰度图变换到调色板图像模式, 预测图像的调色板信息从标准分割图像中提取
color_palette=label_img.getpalette()
img_prediction.putpalette(color_palette)
# get ground truth label image color palette for prediction image

# 5
# Only save output file /仅保存分割后的图像，不需要输出对比图像到此为止
img_prediction.save('output.png')

# 6
# visualization /可视化，输出对比图像依次为：原图、分割标准图像、模型分割后的图像、融合图像
mini_batch = []
# orginal image 原图
img_tensor=T.functional.to_tensor(img)
mini_batch.append(img_tensor)
# ground truth label image 分割标准图像
label_img_tensor = T.functional.to_tensor(label_img.convert('RGB'))
mini_batch.append(label_img_tensor)
# prediction label image 模型分割后的图像
img_prediction_rgb = img_prediction.convert('RGB')
img_prediction_tensor = T.functional.to_tensor(img_prediction_rgb)
mini_batch.append(img_prediction_tensor)
# blending image 融合图像
blend_img = Image.blend(img,img_prediction_rgb,alpha=0.5)
blend_img_tensor = T.functional.to_tensor(blend_img)
mini_batch.append(blend_img_tensor)

# Show images with matplotlib /用matplotlib显示图像
grid_img = vutils.make_grid(mini_batch,padding=3,pad_value=1)
plt.axis('off')
plt.imshow(grid_img.permute(1,2,0))
plt.show()

# Save result image to file /保存对比图像到文件
vutils.save_image(mini_batch,'result2.png')
