from model.googlenet import googleNet
from torchvision import transforms, datasets
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import os
import json


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using device is {}.".format(device))
    data_transform = {"train": transforms.Compose([transforms.RandomResizedCrop(224),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                        std=(0.5, 0.5, 0.5))]),
                      "val": transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                      std=(0.5, 0.5, 0.5))])}

    data_path = os.path.join(os.getcwd(), 'data')
    assert os.path.exists(data_path), "file {} not exist!".format(data_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'train'),
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(data_path, 'val'),
                                       transform=data_transform["val"])
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    print("using {} images training , using {} images val".format(num_train, num_val))
    classes_dict = train_dataset.class_to_idx
    classes_dict = dict((key, value) for value, key in classes_dict.items())

    json_string = json.dumps(classes_dict, indent=5)

    json_path = "./class_dict.json"
    with open(json_path, 'w') as f:
        f.write(json_string)
    batch_size = 32
    nm = min([os.cpu_count(), batch_size if batch_size > 0 else 0, 8])
    print("using num_workers is {}".format(nm))
    # shuffle 操作十分重要，不洗牌 测试集 性能无法得到提升
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=nm)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=nm)

    net = googleNet(AuxFlage=True, num_classes=5, init_weight=True)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0003)

    epochs = 30

    best_acc = 0.0
    save_path = './googleNet.pth'
    train_steps = len(train_loader)
    for epoch in range(epochs):
        all_loss = 0.0
        net.train()
        train_bar = tqdm(train_loader)

        for steps, data in enumerate(train_bar):
            images, labels = data
            res, aux1, aux2 = net(images.to(device))
            loss1 = loss_function(res, labels.to(device))
            loss2 = loss_function(aux1, labels.to(device))
            loss3 = loss_function(aux2, labels.to(device))
            loss = loss1 + 0.3 * loss2 + 0.3 * loss3
            # 先使用optimizer.zero_grad把梯度信息设置为0。
            # 用loss.backward方法时候，
            # Pytorch的autograd就会自动沿着计算图反向传播，
            # optimizer.step用来更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_loss += loss.item()
            train_bar.desc = "train epoch {}/{} loss:{:.3f}.".format(epoch + 1, epochs, loss)

        # val
        acc = 0.0
        net.eval()
        val_bar = tqdm(val_loader)
        with torch.no_grad():

            for data_val in val_bar:
                val_images, val_labels = data_val
                outs = net(val_images.to(device))
                predict = torch.max(outs, 1)[1]
                acc += torch.eq(predict, val_labels.to(device)).sum().item()
        val_acc = acc / num_val

        print("epoch {} train_loss: {:.3f} val_acc:{:.3f}".format(epoch + 1, all_loss / train_steps, val_acc))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), save_path)
    print("Finished !")


if __name__ == "__main__":
    train()
