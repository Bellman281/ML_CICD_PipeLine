import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.transforms import Normalize, ToTensor
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F  
import time
from collections import OrderedDict
import os

n_epochs = 1
batch_size_train = 64
batch_size_test = 32
learning_rate = 0.01
momentum = 0.5
first_HL = 10
model_name =  "EMNIST_SPINALNET_model_22JUL2022.pth"

def check_backend():
    global device
    print(f" Pytorch Version {torch.__version__}")
    print(f" Pytorch Version {torchvision.__version__}")
    print (f' MPS backend is bulit? {torch.backends.mps.is_built()}')
    print( f' MPS backend is available {torch.backends.mps.is_available()}')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f' Device is set to {device}')
    return 


# load data in

def load_data(val_split=0.8):
    
    train_set = datasets.EMNIST(root="data", split="balanced", train=True, 
                                transform = transforms.Compose([ToTensor(),
                               Normalize( (0.1307,), (0.3081,))])  )
                                                    
    test_set = datasets.EMNIST(root="data", split="balanced", train=False, 
                               transform=transforms.Compose([
                               ToTensor(),
                               Normalize((0.1307,), (0.3081,))]) )
    
    train_ = torch.utils.data.DataLoader(train_set, shuffle=True)

    split_ = int(val_split*(len(train_)))  
    valid_ = len(train_) - split_ 

    train_set, val_set = torch.utils.data.random_split(train_set, [split_, valid_]) 

    print(f' train size: {len(train_set)}, val size: {len(val_set)} , test size: {len(test_set)} ')
    classes = test_set.classes
    return train_set, val_set, test_set,classes





check_backend()
train_set, val_set, test_set,classes = load_data(val_split=0.8)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(160, first_HL) #changed from 16 to 8
        self.fc1_1 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_2 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_3 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_4 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc1_5 = nn.Linear(160 + first_HL, first_HL) #added
        self.fc2 = nn.Linear(first_HL*6, 47) # changed first_HL from second_HL
        
   
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x1 = x[:, 0:160]
        
        x1 = F.relu(self.fc1(x1))
        x2= torch.cat([ x[:,160:320], x1], dim=1)
        x2 = F.relu(self.fc1_1(x2))
        x3= torch.cat([ x[:,0:160], x2], dim=1)
        x3 = F.relu(self.fc1_2(x3))
        x4= torch.cat([ x[:,160:320], x3], dim=1)
        x4 = F.relu(self.fc1_3(x4))
        x5= torch.cat([ x[:,0:160], x4], dim=1)
        x5 = F.relu(self.fc1_4(x5))
        x6= torch.cat([ x[:,160:320], x5], dim=1)
        x6 = F.relu(self.fc1_5(x6))

        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        x = torch.cat([x, x5], dim=1)
        x = torch.cat([x, x6], dim=1)
        x = self.fc2(x)
        
    
        return F.log_softmax(x, dim = 1)
    


def train_m(model, optimizer, criterion):
  
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
    
    model.train()
    
    correct = 0
    total = 0
    train_loss = 0
    
    for i, (images, labels) in enumerate(train_loader):
        #3inputs, labels = data
        images = images.to(device)
        labels = labels.to(device)
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()
        
        
        train_loss += loss.item()
        
        _, prediction = torch.max(outputs.data, 1)  
        total += labels.size(0)
        correct += (prediction == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * correct / total
    
    return model, train_loss, train_acc  


def validate_m(model, criterion):

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_train, shuffle=True)
    
    model.eval()

    correct = 0
    total = 0
    val_loss = 0 
    
    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

    return val_loss, val_acc



def test_m(model):

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

    model.eval()
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        test_acc = 100 * correct / total

    return test_acc



def model_explore(n_epochs = 2):
 
    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    time_total = 0
    
    for epoch in range(n_epochs): 
        time_start = time.time()
        model, train_loss, train_acc = train_m(model, optimizer, criterion)
        val_loss, val_acc = validate_m(model, criterion)
        time_end = time.time()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        time_duration = round(time_end - time_start, 2)
        time_total += time_duration
        
      
        print(f'Epoch {epoch},    train acc: {train_acc:6.2f},train loss: {train_loss:7.4f},\t val acc: {val_acc:6.2f}, val loss: {val_loss:7.4f},  \t time: {time_duration}s')
    

    results = OrderedDict()
    results['train_losses'] = [round(x, 4) for x in train_losses]
    results['val_losses'] = [round(x, 4) for x in val_losses]
    results['train_accs'] = [round(x, 2) for x in train_accs]
    results['val_accs'] = [round(x, 2) for x in val_accs]
    results['train_acc'] = round(train_acc, 2)
    results['val_acc'] = round(val_acc, 2)
    results['time_total'] = round(time_total, 2)
    
    return results, model



results,model = model_explore(n_epochs= n_epochs)

test_acc = test_m(model)


print(f'test acc: {round(test_acc, 2)}')
print(f'total training time: {results["time_total"]}')
print()


torch.save(model.state_dict(),model_name)

print(f"model saved as {model_name}")

#loaded_model = Net()
#loaded_model.load_state_dict(torch.load('EMNIST_SPINALNET_model_22JUL2022.pth'))

