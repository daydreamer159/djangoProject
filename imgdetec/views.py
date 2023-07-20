from django.http import JsonResponse, HttpResponse
from django.shortcuts import render

from djangoProject import settings
from imgdetec import models
from model.vit_model import vit_base_patch16_224_in21k as create_model
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json

def imgSave(request):
    if request.method == 'POST':
        new_img = models.ImgInfo(
            img=request.FILES.get('imageUpload'),
            imginfo=request.FILES.get('imageUpload').name,
        )
        new_img.save()
        print("上传成功！")
        return render(request, 'ok.html')
    else:
        return render(request, 'upload.html')


def upload_image(request, context=None):
    # if request.method == 'POST' and request.FILES.get('imageUpload'):
    #     new_img = models.ImgInfo(
    #         img=request.FILES.get('imageUpload'),
    #         imginfo=request.FILES.get('imageUpload').name,
    #     )
    #     new_img.save()
    #     img=request.FILES.get('imageUpload')
    #     with open('../templates/static/img', "wb+") as f:
    #         f.write(img)
    #     print("上传成功！")

    if request.method == 'POST' and request.FILES.get('imageUpload'):
        img_file = request.FILES.get('imageUpload')
        # 使用 os 模块的 join 函数构建文件路径
        filename = os.path.join(settings.BASE_DIR, 'templates', 'static', 'img', img_file.name)
        # 以二进制方式打开文件，使用 with 语句自动关闭文件
        with open(filename, 'wb+') as f:
            f.write(img_file.read())
        print("上传成功！")
        # time.sleep(5)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        img_path = os.path.join('D:/djangoProject/templates/static/img', img_file.name)
        # load image
        # time.sleep(5)
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
        response_data = {
            'message': '图片上传成功',
            'result': res
        }
        context = {'response_data': json.dumps(response_data)}
        return JsonResponse(response_data)

    return render(request, 'upload.html')
