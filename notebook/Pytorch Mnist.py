<<<<<<< HEAD
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_images_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\train-images.idx3-ubyte"
train_labels_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\train-labels.idx1-ubyte"

test_images_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\t10k-images.idx3-ubyte"
test_labels_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\t10k-labels.idx1-ubyte"
x_train = idx2numpy.convert_from_file(train_images_path).reshape(-1,784)
x_train = x_train/255

x_train = torch.tensor(x_train,dtype=torch.float32).to(device)

y_train = idx2numpy.convert_from_file(train_labels_path)
y_train=y_train

y_train = torch.tensor(y_train,dtype=torch.long).to(device)

x_test=idx2numpy.convert_from_file(test_images_path).reshape(-1,784)/255

x_test=torch.tensor(x_test,dtype=torch.float32).to(device)

y_test=idx2numpy.convert_from_file(test_labels_path)

y_test=torch.tensor(y_test,dtype=torch.long).to(device)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.wb1=nn.Linear(784,256)
        self.wb2=nn.Linear(256,10)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.wb1(x))
        x=self.wb2(x)
        return x

model=NN().to(device)
l=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)

for i in range(10000):

    optimizer.zero_grad()
    output = model(x_train)
    loss=l(output,y_train)
    loss.backward()
    optimizer.step()

    if i%100==0:
        print(f"loss{loss.item()}")


i=67

with torch.no_grad():
    pred=model(x_test)
    pred_label=torch.argmax(pred,axis=1)


plt.imshow(x_test[i].cpu().numpy().reshape(28,28),cmap="gray")
plt.title(f"predicted = {pred_label[i].item()},original = {y_test[i].item()}") 
=======
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

train_images_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\train-images.idx3-ubyte"
train_labels_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\train-labels.idx1-ubyte"

test_images_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\t10k-images.idx3-ubyte"
test_labels_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\t10k-labels.idx1-ubyte"
x_train = idx2numpy.convert_from_file(train_images_path).reshape(-1,784)
x_train = x_train/255

x_train = torch.tensor(x_train,dtype=torch.float32).to(device)

y_train = idx2numpy.convert_from_file(train_labels_path)
y_train=y_train

y_train = torch.tensor(y_train,dtype=torch.long).to(device)

x_test=idx2numpy.convert_from_file(test_images_path).reshape(-1,784)/255

x_test=torch.tensor(x_test,dtype=torch.float32).to(device)

y_test=idx2numpy.convert_from_file(test_labels_path)

y_test=torch.tensor(y_test,dtype=torch.long).to(device)


class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.wb1=nn.Linear(784,256)
        self.wb2=nn.Linear(256,10)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.relu(self.wb1(x))
        x=self.wb2(x)
        return x

model=NN().to(device)
l=nn.CrossEntropyLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)

for i in range(10000):

    optimizer.zero_grad()
    output = model(x_train)
    loss=l(output,y_train)
    loss.backward()
    optimizer.step()

    if i%100==0:
        print(f"loss{loss.item()}")


i=67

with torch.no_grad():
    pred=model(x_test)
    pred_label=torch.argmax(pred,axis=1)


plt.imshow(x_test[i].cpu().numpy().reshape(28,28),cmap="gray")
plt.title(f"predicted = {pred_label[i].item()},original = {y_test[i].item()}") 
>>>>>>> 2f0dda925c800bf593adcd857517ad6d95bbbf7f
plt.show()