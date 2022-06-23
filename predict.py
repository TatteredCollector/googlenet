import torch
from torchvision import transforms
from model.googlenet import googleNet
import matplotlib.pyplot as plt
import os
import json
from PIL import Image


def predict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                              std=(0.5, 0.5, 0.5))])
    data_path = "./flowers.jpg"
    assert os.path.exists(data_path), "file {} is not exist!".format(data_path)
    img = Image.open(data_path)
    plt.imshow(img)
    img = data_transform(img).to(device)
    img = torch.unsqueeze(img, dim=0)

    class_dict_path = "./class_dict.json"
    assert os.path.exists(class_dict_path), "file {} is  not exist!".format(class_dict_path)

    with open(class_dict_path, 'r') as f:
        class_dict = json.load(f)

    net = googleNet(AuxFlage=False, num_classes=5, init_weight=False)
    net.to(device)
    net.eval()

    weight_path = "./googleNet.pth"
    assert os.path.exists(weight_path), "file {} is not exist!".format(weight_path)
    # model.load_state_dict(state_dict, strict=False)
    # strict =False忽略找不到的 对应参数 不需要严格执行
    net.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    with torch.no_grad():
        outs = net(img)
        predicts = torch.squeeze(torch.softmax(outs, dim=1)).cpu()
        predict_cla = torch.argmax(predicts).numpy()
    img_til = "class : {} prob :{:.3f} ".format(class_dict[str(predict_cla)],
                                                predicts[predict_cla].numpy())
    plt.title(img_til)
    plt.show()

    for i in range(len(predicts)):
        print("class: {} prob:{}".format(class_dict[str(i)], predicts[i].numpy()))


if __name__ == "__main__":
    predict()
