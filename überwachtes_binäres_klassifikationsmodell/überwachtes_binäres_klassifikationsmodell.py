import numpy as np
import json

# json
with open("trainingdataset.json", "r") as f:
    data = json.load(f)
trainingdata = np.array(data.get("trainingdata", []))
labels = np.array(data.get("aim_data", None))


#------------------------------
## input to hidden w and b matrix 3 x 4
weight0 = np.random.rand(100,10)
bias0 = np.random.rand(10)


weight1 = np.random.rand(10,1) # Auf eine Zufallszahl gesetzt.
bias1 = np.random.rand(1)   # Auf eine Zufallszahl gesetzt.




def relu_function(x):
    return np.maximum(0, x)

def relu_function_derivative(x):
    return (x > 0).astype(float)

def sigmoid_function(x):
    return 1/(1 + np.exp(-x))

def sigmoid_function_derivative(x):
    return x * (1 - x)

def binary_log_loss_function(acc,label):
    acc = np.clip(acc, 1e-12, 1 - 1e-12)
    return label * np.log(acc) + (1 - label) * np.log(1 - acc)


def sigmoid_d_X_log_loss_d_function(acc,label):    #sigmoid_derivative_multiplied_with_binary_log_loss_derivative_function
    return acc - label                                                            

def cost_function(loss):
    return -np.mean(loss)

epochs = 5000
learn_rate = 0.001

for epoch in range(epochs):

    indices = np.random.permutation(len(trainingdata))

    for i in indices:

        acc0 = trainingdata[i]

        z1 =  np.dot(acc0 , weight0) + bias0
        acc1 = relu_function(z1)

        z2 =  np.dot(acc1 , weight1) + bias1
        acc2 = sigmoid_function(z2)


        #bakpropagation
        label = labels[i][0]

        weight0 -=  learn_rate * np.outer(acc0,((weight1.reshape(10,) * relu_function_derivative(z1)* sigmoid_d_X_log_loss_d_function(acc2,label))))
        bias0   -=  learn_rate * (weight1.reshape(10,) * relu_function_derivative(z1)* sigmoid_d_X_log_loss_d_function(acc2,label))

        weight1 -= learn_rate *  (acc1.reshape(10,1) * sigmoid_d_X_log_loss_d_function(acc2,label))
        bias1   -= learn_rate *  sigmoid_d_X_log_loss_d_function(acc2,label)
        
        


        
def testing(data):
    z1 = np.dot(data, weight0) + bias0
    a1 = relu_function(z1)

    z2 = np.dot(a1, weight1) + bias1
    a2 = sigmoid_function(z2)

    print("Output:", a2)

#testing(trainingdata[0]) # 1
#testing(trainingdata[6]) # 0
#testing(trainingdata[7]) #1 
#testing(trainingdata[4]) #0


def save_network_params(weight1, weight2, bias1, bias2, file_path="bias_and_weights.json"):
    # NumPy-Arrays in Listen umwandeln
    data = {
        "weight1": weight1.tolist(),
        "weight2": weight2.tolist(),
        "bias1": bias1.tolist(),
        "bias2": bias2.tolist()
    }

    # In die JSON-Datei schreiben (überschreibt alte Daten)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


save_network_params(weight0, weight1, bias0, bias1)

print("done")



