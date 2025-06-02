from tinygrad import Tensor, nn
import csv
import numpy as np


class MLP:
    def __init__(self):
        self.l1 = Tensor.kaiming_uniform(38, 10)
        # self.l2 = Tensor.kaiming_uniform(3, 10)

    def __call__(self, x: Tensor):
        # First layer
        with open("data/test.csv", mode="w") as csvfile:
            writer = csv.writer(csvfile)
            for i in range(120, 158): # analyze which positions in 784 most often have digits and pick these (contiguous) to train model on
                t = np.frombuffer(x[0][i].data(), dtype=np.float32)
                writer.writerow([t])
        x = x.flatten(1).dot(self.l1)
        # x = x + self.b1
        # x = x.relu()

        # Second layer
        # x = x.dot(self.l2)
        # x = x + self.b2
        return x


def debug_forward(self, x: Tensor):
    # First layer
    print(x.flatten(1).numpy())
    x1 = x.flatten(1).dot(self.l1)
    print("After first linear:", x1[0, :5].numpy())
    return x1
    # x2 = x1 #+ self.b1
    # print("After first bias:", x2[0, :5].numpy())
    # x3 = x2.relu()
    # print("After relu:", x3[0, :5].numpy())

    # # Second layer
    # x4 = x3.dot(self.l2)
    # print("After second linear:", x4[0, :5].numpy())
    # x5 = x4 #+ self.b2
    # print("Final activations:", x5[0].numpy())
    # return x5


model = MLP()
optim = nn.optim.Adam([model.l1], lr=0.001)


# train
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
        x += [list(map(lambda x: float(x) / 255.0, row[1:]))]

x = Tensor(x)
y = Tensor(y)

with Tensor.train():
    for i in range(300):
        optim.zero_grad()
        loss = model(x).sparse_categorical_crossentropy(y).backward()
        optim.step()
        print(i, loss.item())

print("Weights for first hidden neuron:", model.l1[:, 0].numpy())


state_dict = nn.state.get_state_dict(model)
l1 = np.frombuffer(state_dict.get("l1").data(), dtype=np.float32)
print("First few weights from l1:", l1[:10])
# print("l1 shape:", l1.reshape(784, 128).shape)
# b1 = np.frombuffer(state_dict.get("b1").data(), dtype=np.float32)
# print("First few weights from b1:", b1[:10])
# l2 = np.frombuffer(state_dict.get("l2").data(), dtype=np.float16)
# print("First few weights from l2:", l2[:10])
# b2 = np.frombuffer(state_dict.get("b2").data(), dtype=np.float32)
# print("First few weights from b2:", b2[:10])

print(f"l1.size {l1.size}")

with open("data/model_weights.csv", mode="w") as csvfile:
    writer = csv.writer(csvfile)
    for i in range(l1.size):
        writer.writerow([l1[i]])
    # for i in range(b1.size):
        # writer.writerow([b1[i]])
    # for i in range(l2.size):
    #     writer.writerow([l2[i]])
    # for i in range(b2.size):
        # writer.writerow([b2[i]])

# run
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

        x = Tensor([list(map(lambda x: float(x) / 255.0, row[1:]))])
        debug_forward(model, x[0:1])
        y_pred = model(x)
        print(np.argmax(np.frombuffer(y_pred.data(), dtype=np.float32)), int(row[0]))
