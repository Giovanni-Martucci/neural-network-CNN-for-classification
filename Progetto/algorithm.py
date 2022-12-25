import cv2
import os
import Progetto
import torch
from PIL import Image
from torchvision import models,transforms
import numpy as np
import argparse
import io
import requests
from torch.autograd import Variable
from torch.nn import functional as F
import pdb
from torch import topk
import skimage.transform
from matplotlib.pyplot import imshow


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="video selected")
args = vars(ap.parse_args())



classes = {
  "0": "Center",
  "1": "Left",
  "2": "Right",
  "3": "Null"
}

for i in os.listdir("./output/"):
    try:
        os.remove('./output/'+i)
    except:
        pass
    

def convert_video(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


vidcap = cv2.VideoCapture("./uploads/"+args["video"])
count = 0
success,image = vidcap.read()
success = True

    
#image = Image.open("./uploads/prova4.png")
image2 = image[...,::-1]
image2 = Image.fromarray(image2)
image2 = test_transform(image2)
img_w= int(image.shape[1])
img_h= int(image.shape[0])
fps = 10.0
#delay =int(1000*1/fps)
out = cv2.VideoWriter("./output/output-"+args["video"][:-4]+".avi",cv2.VideoWriter_fourcc(*'MJPG'), fps, (img_w,img_h),True)
model = Progetto.get_model()
model.load_state_dict(torch.load('./weights/resnet_dataset_finetuning-99.pth',map_location=torch.device('cpu')))
model.eval()



####################### CAM #####################
# # normalize = transforms.Normalize(
# #    mean=[0.485, 0.456, 0.406],
# #    std=[0.229, 0.224, 0.225]
# # )
# # preprocess = transforms.Compose([
# #    transforms.Resize((224,224)),
# #    transforms.ToTensor(),
# #    normalize
# # ])

# # features_blobs = []
# # def hook_feature(module, input, output):
# #     features_blobs.append(output.data.cpu().numpy())

# # model._modules.get('layer4').register_forward_hook(hook_feature)

# # # get the softmax weight
# # params = list(model.parameters())
# # weight_softmax = np.squeeze(params[-2].data.numpy())

# # def returnCAM(feature_conv, weight_softmax, class_idx):
# #     # generate the class activation maps upsample to 256x256
# #     size_upsample = (256, 256)
# #     bz, nc, h, w = feature_conv.shape
# #     output_cam = []
# #     for idx in class_idx:
# #         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
# #         cam = cam.reshape(h, w)
# #         cam = cam - np.min(cam)
# #         cam_img = cam / np.max(cam)
# #         cam_img = np.uint8(255 * cam_img)
# #         output_cam.append(cv2.resize(cam_img, size_upsample))
# #     return output_cam

# # response = requests.get(image)
# # img_pil = Image.open(io.BytesIO(response.content))
# # img_pil.save('test.jpg')

# # img_tensor = preprocess(img_pil)
# # img_variable = Variable(img_tensor.unsqueeze(0))
# # logit = model(img_variable)



# # h_x = F.softmax(logit, dim=1).data.squeeze()
# # probs, idx = h_x.sort(0, True)
# # probs = probs.numpy()
# # idx = idx.numpy()

# # # output the prediction
# # for i in range(0, 5):
# #     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# # # generate class activation mapping for the top1 prediction
# # CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# # # render the CAM and output
# # print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
# # img = cv2.imread('test.jpg')
# # height, width, _ = img.shape
# # heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
# # result = heatmap * 0.3 + img * 0.5
# # cv2.imwrite('CAM.jpg', result)
# image = Image.open("image.jpg").convert('RGB')
# imshow(image)

# normalize = transforms.Normalize(
#    mean=[0.485, 0.456, 0.406],
#    std=[0.229, 0.224, 0.225]
# )

# # Preprocessing - scale to 224x224 for model, convert to tensor, 
# # and normalize to -1..1 with mean/std for ImageNet

# preprocess = transforms.Compose([
#    transforms.Resize((224,224)),
#    transforms.ToTensor(),
#    normalize
# ])

# display_transform = transforms.Compose([
#    transforms.Resize((224,224))])


# tensor = preprocess(image)
# prediction_var = Variable((tensor.unsqueeze(0)).cuda(), requires_grad=True)


# class SaveFeatures():
#     features=None
#     def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
#     def hook_fn(self, module, input, output): self.features = ((output.cpu()).data).numpy()
#     def remove(self): self.hook.remove()

# final_layer = model._modules.get('layer4')
# activated_features = SaveFeatures(final_layer)

# prediction = model(prediction_var)
# pred_probabilities = F.softmax(prediction).data.squeeze()
# activated_features.remove()
# topk(pred_probabilities,1)

# def getCAM(feature_conv, weight_fc, class_idx):
#     _, nc, h, w = feature_conv.shape
#     cam = weight_fc[class_idx].dot(feature_conv.reshape((nc, h*w)))
#     cam = cam.reshape(h, w)
#     cam = cam - np.min(cam)
#     cam_img = cam / np.max(cam)
#     return [cam_img]

# weight_softmax_params = list(model._modules.get('fc').parameters())
# weight_softmax = np.squeeze(weight_softmax_params[0].cpu().data.numpy())
# weight_softmax_params
# class_idx = topk(pred_probabilities,1)[1].int()
# overlay = getCAM(activated_features.features, weight_softmax, class_idx )
# imshow(overlay[0], alpha=0.5, cmap='jet')
# imshow(display_transform(image))
# imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')
# # class_idx = topk(pred_probabilities,2)[1].int()
# # overlay = getCAM(activated_features.features, weight_softmax, 332 )
# # imshow(display_transform(image))
# # imshow(skimage.transform.resize(overlay[0], tensor.shape[1:3]), alpha=0.5, cmap='jet')


####################### CAM #####################

while success:   
    output = model(image2.unsqueeze(0))
    output2 = output.tolist()
    output2 = np.reshape(output2,-1)
    m = max(output2)
    index = [i for i, j in enumerate(output2) if j == m]
    print(index[0])
    cv2.putText(image, str(classes[str(index[0])]), (50,100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
    out.write(image)
      
    
    success,image = vidcap.read()
    if success:
        image2 = image[...,::-1]
        image2 = Image.fromarray(image2)
        image2 = test_transform(image2)
    count += 1
    
out.release()
convert_video('./output/output-'+args["video"][:-4]+'.avi','output/final_output-'+args["video"][:-4])
#uso un converitore da avi a mp4 perche se codifico direttamente in mp4 viene troppo di bassa qualità
os.remove('./uploads/'+args["video"])
    