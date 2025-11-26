import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivation(x):
    return x*(1-x)
      


x=np.array([[10],[20],[30],[40],[50]])
y=np.array([[1],[2],[3],[4],[5]])

np.random.seed(42)

hidden_weights=np.random.randn(1,2)*0.01
hidden_bias=np.random.randn(1,2)*0.01
output_weights=np.random.randn(2,1)*0.01
output_bias=np.random.randn(1,1)*0.01

learning_rate=0.001

print(x.shape,y.shape,hidden_weights.shape,hidden_bias.shape,output_weights.shape,output_bias.shape)
for i in range(100000):
    hidden_output=sigmoid(np.dot(x,hidden_weights)+hidden_bias)
   
    final_output=sigmoid(np.dot(hidden_output,output_weights)+output_bias)
    
    output_error=(y-final_output)*sigmoid_derivation(final_output)
    hidden_error=np.dot(output_error,output_weights.T)*sigmoid_derivation(hidden_output)

    hidden_weights +=learning_rate*np.dot(x.T,hidden_error)
    hidden_bias += learning_rate*np.sum(hidden_error,axis=0,keepdims=True)
    output_weights += learning_rate*np.dot(hidden_output.T,output_error)
    output_bias  += learning_rate*np.sum(output_error,axis=0,keepdims=True)

a=np.array([[30]])
hidden_output=sigmoid(np.dot(a,hidden_weights)+hidden_bias)
final_output=sigmoid(np.dot(hidden_output,output_weights)+output_bias)

predicted_index=np.argmax(final_output)

prediction=y[predicted_index]
print(prediction)