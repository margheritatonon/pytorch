import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#this creates a tensor
x = torch.rand(1, 2)
print(x)

#training data from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
print("----------")

#test data from open datasets
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64 #number of samples processed in one batch

#these load the data in batches
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader: 
    print(f"Shape of X [N, C, H, W]: {X.shape}") #Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    print(f"Shape of y: {y.shape} {y.dtype}") #Shape of y: torch.Size([64]) torch.int64
    break

#defining a neural network
#we create a class that inherits from nn.Module
#in the __init__ function: define the layers of the network
#forward function: define how data will pass through the network

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#print(f"Using {device} device") #using mps device

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__() #inherit from nn.modules
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        ) #this defines the layers
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#to train a model: need loss function and optimizer
loss_fn = nn.CrossEntropyLoss() #cross entropy as the loss (classification problem)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #stochastic gradient descent

#In a single training loop, the model makes prediction on training data and backpropagates the error to adjust params

def train(dataloader, model, loss_fn, optimizer):
    """
    Trains a neural network
    """
    size = len(dataloader.dataset)
    model.train() #trainng mode
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) #move data to device

        #prediction error
        pred = model(X) #model uses input data to make predictions
        loss = loss_fn(pred, y) #we calculate the loss (cross entropy) for predictions

        #backprop
        loss.backward()
        optimizer.step() #performs a single optimization step - does backpropagation
        optimizer.zero_grad() #zeroes out the gradients so they dont accumulate

        if batch % 100 == 0: #we do this to visualize how the training process is going
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#testing!
def test(dataloader, model, loss_fn):
    """
    Check the model's performance against the test set.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() #evaluation mode
    test_loss, correct = 0, 0 #initializing the loss and the correctly classified examples
    with torch.no_grad(): #disables gradient calculation - we don't need this during testing
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item() #adds to the loss of the model (for incorrect classifications)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #add the correctly classified to correct
    test_loss /= num_batches #average the loss and the accuracy (correct)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#running everything:
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#we have 64.6% accuracy

#saving the model:
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth") #now we have it saved here

#reloading a model:
#model = NeuralNetwork().to(device)
#model.load_state_dict(torch.load("model.pth", weights_only=True))

#using the model to make predictions:
classes = [ #these are the names of the actual classes in the dataset
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() #switch the model to evaluation mode
x, y = test_data[0][0], test_data[0][1] #select a single test sample
with torch.no_grad(): #inference with no gradients
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')