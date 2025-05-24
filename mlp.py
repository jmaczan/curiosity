from tinygrad import Tensor, nn
import csv
import numpy as np


class MLP:
    def __init__(self):
        self.l1 = Tensor.kaiming_uniform(784, 128)
        self.b1 = Tensor.zeros(128)
        self.l2 = Tensor.kaiming_uniform(128, 10)
        self.b2 = Tensor.zeros(10)

    def __call__(self, x: Tensor):
        x = x.flatten(1).dot(self.l1) + self.b1
        x = x.relu()
        x = x.dot(self.l2) + self.b2
        return x


model = MLP()
optim = nn.optim.Adam([model.l1, model.b1, model.l2, model.b2], lr=0.001)


with open("data/mnist_train.csv") as csvfile:
    reader = csv.reader(csvfile)
    x = []
    y = []
    for index, row in enumerate(reader):
        if index == 0:
            continue
        if index > 3000:
            break
        y.append(int(row[0]))
        x += [list(map(lambda x: int(x), row[1:]))]

x = Tensor(x)
y = Tensor(y)

with Tensor.train():
    for i in range(3):
        optim.zero_grad()
        loss = model(x).sparse_categorical_crossentropy(y).backward()
        optim.step()
        print(i, loss.item())

state_dict = nn.state.get_state_dict(model)
l1 = np.frombuffer(state_dict.get("l1").data(), dtype=np.float32)
b1 = np.frombuffer(state_dict.get("b1").data(), dtype=np.float32)
l2 = np.frombuffer(state_dict.get("l2").data(), dtype=np.float32)
b2 = np.frombuffer(state_dict.get("b2").data(), dtype=np.float32)

print(f"l1.size {l1.size}")

with open("data/model_weights.csv", mode="w") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(l1.size):
        writer.writerow([l1[i]])
    for i in range(b1.size):
        writer.writerow([b1[i]])
    for i in range(l2.size):
        writer.writerow([l2[i]])
    for i in range(b2.size):
        writer.writerow([b2[i]])

with open("data/mnist_train.csv") as csvfile:
    reader = csv.reader(csvfile)
    x = []
    y = []
    print("y_pred, y")
    for index, row in enumerate(reader):
        if index == 0 or index <= 3000:
            continue
        if index > 3010:
            break

        raw = list(map(lambda x: int(x), row[1:]))
        y_pred = model(Tensor([list(map(lambda x: int(x), row[1:]))]))
        print(np.argmax(np.frombuffer(y_pred.data(), dtype=np.float32)), int(row[0]))
