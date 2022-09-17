import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from model import TeacherModel, StudentModel

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 载入数据集 使用MNIST跑一个demo
train_dataset = torchvision.datasets.MNIST(
    root='dataset/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='dataset/',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# 训练teacher model
teacher_model = TeacherModel()
teacher_model = teacher_model.to(device)
print(summary(teacher_model))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(teacher_model.parameters(), lr=1e-4)

for epoch in range(3):
    teacher_model.train()

    for X, labels in tqdm(train_dataloader):
        X = X.to(device)
        labels = labels.to(device)

        # forward
        output = teacher_model(X)
        loss = criterion(output, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    teacher_model.eval()
    correct_num = 0
    total_num = 0

    with torch.no_grad():
        for X, labels in tqdm(test_dataloader):
            X = X.to(device)
            labels = labels.to(device)

            output = teacher_model(X)
            predictions = output.max(1).indices
            correct_num += (predictions == labels).sum()
            total_num += predictions.size(0)
        acc = (correct_num/total_num).item()

    teacher_model.train()
    print('Teacher Model: Epoch:{}  Accuracy:{:.4f}'.format(epoch+1, acc))


student_model = StudentModel()
student_model = student_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)


# 使用传统方法训练student model
for epoch in range(3):

    student_model.train()

    for X, y in tqdm(train_dataloader):
        X = X.to(device)
        y = y.to(device)

        # forward
        output = student_model(X)
        loss = criterion(output, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student_model.eval()
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            X = X.to(device)
            y = y.to(device)

            output = student_model(X)
            predictions = output.max(1).indices
            correct_num += (predictions == y).sum()
            total_num += predictions.size(0)
    acc = correct_num/total_num
    student_model.train()

    print('Student Model: Epoch:{}  Accuracy:{:.4f}'.format(epoch+1, acc))


# 使用知识蒸馏训练student model
teacher_model.eval()
student_model_KD = StudentModel()
student_model_KD = student_model_KD.to(device)

optimizer = torch.optim.Adam(student_model_KD.parameters(), lr=1e-4)
temp = 7

# hard loss与hard loss的权重
hard_criterion = nn.CrossEntropyLoss()
alpha = 0.3

# soft loss
soft_criterion = nn.KLDivLoss(reduction='batchmean')

for epoch in range(3):
    student_model_KD.train()
    for X, y in tqdm(train_dataloader):
        X = X.to(device)
        y = y.to(device)

        # 从teacher model的预测中获取soft label
        with torch.no_grad():
            soft_label = teacher_model(X)

        output = student_model_KD(X)
        # 用student model的输出与ground_truth计算得出hard loss
        hard_loss = hard_criterion(output, y)
        # 用student model的输出与teacher model的输出计算soft loss
        soft_loss = soft_criterion(
            F.softmax(output / temp, dim=1),
            F.softmax(soft_label / temp, dim=1)
        )

        # forward
        loss = alpha * hard_loss + (1 - alpha) * soft_loss

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    student_model_KD.eval()
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        for X, y in tqdm(test_dataloader):
            X = X.to(device)
            y = y.to(device)

            output = student_model_KD(X)
            predictions = output.max(1).indices
            correct_num += (predictions == y).sum()
            total_num += predictions.size(0)
    acc = correct_num/total_num
    student_model_KD.train()

    print('Student Model using Knowledge Distilling: Epoch:{}  Accuracy:{:.4f}'.format(epoch + 1, acc))

