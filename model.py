import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherModel(nn.Module):
    def __init__(self, class_num=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, class_num)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


class StudentModel(nn.Module):
    def __init__(self, class_num=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, class_num)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)

        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        # x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


