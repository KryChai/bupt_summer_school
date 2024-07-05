from matplotlib import pyplot as plt
from torchmetrics.functional import precision, recall, f1_score

from unet_model import UNet
from dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
# 创建一个列表来存储损失值
loss_values = []
# 创建一个字典来存储评估指标值
metrics_history = {'precision': [], 'recall': [], 'f1': []}

def train_net(net, device, data_path, epochs=40, batch_size=8, lr=0.00001):
    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    # 训练epochs次
    for epoch in range(epochs):
        count = 0
        # 训练模式
        net.train()
        ls = []
        p = []
        r = []
        f = []
        # 按照batch_size开始训练
        for image, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            # 使用网络参数，输出预测结果
            pred = net(image)
            # # 将预测结果转换为二值图像
            pred_binary = (pred > 0.5).float()
            # 计算loss
            loss = criterion(pred, label)
            ls.append(loss.item())
            # np.append(ls, loss.item())
            # ls.append(loss.item())
            # print(f'epoch {epoch} Loss/train', loss.item())
            # 计算P值、R值和F1分数
            # 计算精度

            precision_gpu = precision(label, pred_binary, task="binary")
            # np.append(p, precision_gpu.item())
            p.append(precision_gpu.item())

            # 计算召回率
            recall_gpu = recall(label, pred_binary, num_classes=1, task="binary")
            # np.append(r, recall_gpu.item())
            r.append(recall_gpu.item())
            # 计算 F1 分数
            f1_gpu = f1_score(label, pred_binary, task="binary", num_classes=1)
            # np.append(f, f1_gpu.item())
            f.append(f1_gpu.item())

            # 计算loss
            loss = criterion(pred, label)
            # loss_values.append(loss.item())
            # print(f'epoch {epoch} Loss/train', loss.item())
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model_150.pth')
            # 更新参数
            loss.backward()
            optimizer.step()

        print(f'epoch {epoch} Loss/train', sum(ls) * 1.0 / len(ls))
        loss_values.append(sum(ls) / len(ls))
        # np.append(loss_values, ls.mean())
        metrics_history['precision'].append(sum(p) / len(p))
        metrics_history['recall'].append(sum(r) / len(r))
        metrics_history['f1'].append(sum(f) / len(f))
        # np.append(metrics_history['precision'], p.mean())
        # np.append(metrics_history['recall'], r.mean())
        # np.append(metrics_history['f1'], f.mean())
        # loss_values.append(ls.mean())
        # metrics_history['precision'].append(p.mean())
        # metrics_history['recall'].append(r.mean())
        # metrics_history['f1'].append(f.mean)

        # 绘制损失值变化图
    print(loss_values)
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    # 保存图表
    plt.savefig('training_loss_200.png')
    print('Loss plot saved as "training_loss.png"')

    # 绘制评估指标变化图
    plt.figure(figsize=(12, 4))
    print(metrics_history['precision'])

    plt.subplot(1, 3, 1)
    plt.plot(metrics_history['precision'], label='Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.title('Precision per Epoch')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(metrics_history['recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall per Epoch')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(metrics_history['f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Epoch')
    plt.legend()

    plt.tight_layout()
    # plt.show()

    # 保存图表
    plt.savefig('evaluation_metrics_200.png')
    print('Evaluation metrics plot saved as evaluation_metrics.png')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道1，分类为1。
    net = UNet(n_channels=1, n_classes=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path = "/tmp/No_Attention/XCAD/train/images/"
    train_net(net, device, data_path,150,8)