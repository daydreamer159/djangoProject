# Create your views here.
from django.shortcuts import render
from app.models import Person
from model.vit_model import vit_base_patch16_224_in21k as create_model
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

# model = torch.load('/test/test/weights/best_model.pth')

def index(request):
    # 查询出Person对象信息，也就是数据表中的所有数据
    # 一行数据就是一个对象，一个格子的数据就是一个对象的一个属性值
    objs = Person.objects.all()

    # locals函数可以将该函数中出现过的所有变量传入到展示页面中，即index.html文件中
    return render(request, 'index.html', locals())

from django.shortcuts import render
from django.http import JsonResponse

# create model
# model = create_model(num_classes=8, has_logits=False).to(device)
# # load model weights
# model_weight_path = "./weights/model-49.pth"
# model.load_state_dict(torch.load(model_weight_path, map_location=device))
# model.eval()
img_path='./test/test/SKIN/AK/ISIC_0024468.jpg'


def upload_image(request):
    if request.method == 'POST' and request.FILES.get('imageUpload'):
        image = request.FILES['imageUpload']
        # 处理上传的图片，例如保存到服务器或进行其他操作
        # ...

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

        img = Image.open(img_path)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './model/class_indices.json'

        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = create_model(num_classes=8, has_logits=False).to(device)
        # load model weights
        model_weight_path = "./model/weights/model-49.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())

        max = 0
        k = 0
        for i in range(len(predict)):
            if predict[i].numpy() > max:
                max = predict[i].numpy()
                k = i
        if class_indict[str(k)] == "VASC":
            class_indict[str(k)] = '血管病变(VASC)'
        if class_indict[str(k)] == "SCC":
            class_indict[str(k)] = '鳞状细胞癌(SCC)'
        if class_indict[str(k)] == "AK":
            class_indict[str(k)] = '日光性角化病(AK)'
        if class_indict[str(k)] == "BCC":
            class_indict[str(k)] = '基底细胞癌(BCC)'
        if class_indict[str(k)] == "BKL":
            class_indict[str(k)] = '良心角化病(BKL)'
        if class_indict[str(k)] == "DF":
            class_indict[str(k)] = '皮肤纤维瘤(DF)'
        if class_indict[str(k)] == "MEL":
            class_indict[str(k)] = '黑色素瘤(MEL)'
        if class_indict[str(k)] == "NV":
            class_indict[str(k)] = '黑素细胞痣(NV)'
        res = "类别：{:10} 置信度：{:.3}".format(class_indict[str(k)], predict[k].numpy())
        # 返回响应
        print(res)
        response_data = {'message': '图片上传成功'}
        return JsonResponse(response_data)

    return render(request, 'upload.html')


# def predict(request):
#     if request.method == 'POST' and request.FILES.get('imageUpload'):
#         # 读取上传的图像文件
#         uploaded_file = request.FILES['image']
#         img = Image.open(uploaded_file)
#
#         # 转换图像为PyTorch格式
#         img_tensor = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#             ])(img).unsqueeze(0)
#
#         # 载入已经训练好的PyTorch模型
#         # model = torch.load('./model/weight/best_model.pth')
#
#         # 在模型中预测结果
#         with torch.no_grad():
#             output = model(img_tensor)
#         prediction = torch.argmax(output, 1)
#
#         # 将预测结果作为变量传递给模板
#         context = {'prediction':prediction.item()}
#         return render(request, 'upload.html', context)
#     else:
#         return render(request, 'upload.html')
#
def imgtest(request):
    if request.method == 'POST' and request.FILES.get('imageUpload'):
    # 读取上传的图像文件
        uploaded_file = request.FILES['image']
        img = Image.open(uploaded_file)

        # 转换图像为PyTorch格式
        img_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])(img).unsqueeze(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # load image
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        with open(json_path, "r") as f:
            class_indict = json.load(f)

        # create model
        model = create_model(num_classes=8, has_logits=False).to(device)
        # load model weights
        model_weight_path = "./weights/model-49.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())

        max = 0
        k = 0
        for i in range(len(predict)):
            if predict[i].numpy() > max:
                max = predict[i].numpy()
                k = i
        if class_indict[str(k)] == "VASC":
            class_indict[str(k)] = '血管病变(VASC)'
        if class_indict[str(k)] == "SCC":
            class_indict[str(k)] = '鳞状细胞癌(SCC)'
        if class_indict[str(k)] == "AK":
            class_indict[str(k)] = '日光性角化病(AK)'
        if class_indict[str(k)] == "BCC":
            class_indict[str(k)] = '基底细胞癌(BCC)'
        if class_indict[str(k)] == "BKL":
            class_indict[str(k)] = '良心角化病(BKL)'
        if class_indict[str(k)] == "DF":
            class_indict[str(k)] = '皮肤纤维瘤(DF)'
        if class_indict[str(k)] == "MEL":
            class_indict[str(k)] = '黑色素瘤(MEL)'
        if class_indict[str(k)] == "NV":
            class_indict[str(k)] = '黑素细胞痣(NV)'
        res = "类别：{:10} 置信度：{:.3}".format(class_indict[str(k)], predict[k].numpy())
        print(res)
        return res