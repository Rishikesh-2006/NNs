import idx2numpy
import numpy as np
import matplotlib.pyplot as plt

def to_categorical(y,num):
    return np.eye(num)[y]

def softmax(a):
    exp_a=np.exp(a-np.max(a,axis=1,keepdims=True))
    return exp_a/np.sum(exp_a,axis=1,keepdims=True)

def sigmoid(a):
    return 1/(1+np.exp(-a))
def relu(a):
    return np.maximum(0,a)
def sigmoid_derivative(a):
    return a*(1-a)
def relu_derivative(a):
    return (a>0).astype(float)


train_images_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\train-images.idx3-ubyte"
train_labels_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\train-labels.idx1-ubyte"

test_images_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\t10k-images.idx3-ubyte"
test_labels_path=r"C:\Users\sarma\OneDrive\Desktop\mnist\t10k-labels.idx1-ubyte"
x_train = idx2numpy.convert_from_file(train_images_path).reshape(-1,784)
x_train = x_train[:10000]/255
y_train=idx2numpy.convert_from_file(train_labels_path)
y_train=y_train[:10000]
y_train=to_categorical(y_train,10)

x_test=idx2numpy.convert_from_file(test_images_path).reshape(-1,784)/255
y_test=idx2numpy.convert_from_file(test_labels_path)
y_test=to_categorical(y_test,10)
print(x_train.shape)

input=784
hidden=256
output=10
np.random.seed(0)
w1=np.random.randn(input,hidden)*np.sqrt(2/input)
b1=np.zeros((1,hidden))
w2=np.random.randn(hidden,output)*np.sqrt(2/output)
b2=np.zeros((1,output))

n=0.001

for i in range(1000):

    a1=np.dot(x_train,w1)+b1
    h=relu(a1)
    a2=np.dot(h,w2)+b2
    y=softmax(a2)

    loss=np.mean((y-y_train)**2)

    if i%10==0:
        print(f" loss={loss*1000: 8f}")
    dly2=(y-y_train)
    dlw2=np.dot(h.T,dly2)/len(h)
    dlb2=np.mean(dly2,axis=0,keepdims=True)

    dly1=np.dot(dly2,w2.T)*relu_derivative(a1)
    dlw1=np.dot(x_train.T,dly1)/len(x_train)
    dlb1=np.mean(dly1,axis=0,keepdims=True)

    w1-=n*dlw1
    w2-=n*dlw2

    b1-=n*dlb1
    b2-=n*dlb2

#plt.show()
#print(f"w1={w1},w2={w2},b1={b1},b2={b2}")

a1=np.dot(x_test,w1)+b1
h=relu(a1)
a2=np.dot(h,w2)+b2
y_pred= np.exp(a2)/np.sum(np.exp(a2),axis=1, keepdims=True)
predicted_labels=np.argmax(y_pred,axis=1)
true_labels= np.argmax(y_test,axis=1)
index=6
plt.imshow(x_test[index].reshape(28,28),cmap="gray")
plt.title(f"predicted={predicted_labels[index]}, Actual={true_labels[index]}")
plt.show()
