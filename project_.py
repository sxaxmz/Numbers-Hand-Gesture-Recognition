# Import Libraries
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Transformation
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Data
train_set = datasets.ImageFolder("data/train", transform = transformations)
val_set = datasets.ImageFolder("data/test", transform = transformations)

# Directory 
dir_name = 'data/train'
folders_list = os.listdir(dir_name)
class_path_list= [os.path.join(dir_name, path) for path in folders_list]

# Get Classes Name
class_name = []
for element in class_path_list:
    head_path, name = os.path.split(element)
    class_name.append(name)

print("class_name: ",class_name)

# DataLoader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)

# Initialize Pretrained Model 
model = models.densenet161(pretrained=True)

# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

# Model Custom Classifier 
classifier_input = model.classifier.in_features
num_labels = len(class_name)
classifier = nn.Sequential(
                           nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, num_labels),
                           nn.LogSoftmax(dim=1)
                           )

model.classifier = classifier

# Specify Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Error/Loss Function
criterion = nn.NLLLoss()

# Optimizer
optimizer = optim.Adam(model.classifier.parameters())

# Trackers
epochs = 20
train_loss_tracker = []
val_loss_tracker = []
train_acc_tracker = []
val_acc_tracker = []
train_class_tracker = []
val_class_tracker = []
train_counter = []
val_counter = []

for epoch in range(epochs):
    train_loss = 0
    val_loss = 0
    accuracy = 0
    
    # Training the model
    model.train()
    print("Training Active ({}/{})".format(epoch, epochs))
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Clear optimizers
        optimizer.zero_grad()

        # Forward pass
        output = model.forward(inputs)

        # Loss
        loss = criterion(output, labels)

        # Calculate gradients (backpropogation)
        loss.backward()

        # Adjust parameters based on gradients
        optimizer.step()

        # Add the loss to the training set's rnning loss
        epoch_loss = loss.item()*inputs.size(0)
        train_loss_tracker.append(epoch_loss)
        train_loss += epoch_loss
        
        # Print the progress of our training
        counter += 1
        #train_counter.append(((epoch-1)*len(train_loader.dataset)))
        train_counter.append(counter)
        print("Training Progress ({}/{})".format(counter, len(train_loader)), "Loss: {:.6f}".format(epoch_loss))

    train_loss = train_loss/len(train_loader.dataset)    
    print("Traiining Step Average Loss: {:.6f} \n".format(train_loss))

    # Evaluating the model
    model.eval()
    print("Evaluation Active")
    counter = 0

    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            output = model.forward(inputs)

            # Calculate Loss
            valloss = criterion(output, labels)

            # Add loss to the validation set's running loss
            epoch_loss = valloss.item()*inputs.size(0)
            val_loss_tracker.append(epoch_loss)
            val_loss += epoch_loss
            
            # Since our model outputs a LogSoftmax, find the real percentages by reversing the log function
            output = torch.exp(output)

            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)

            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)


            # Calculate the mean (get the accuracy for this batch) and add it to the running accuracy for this epoch
            epoch_acc = torch.mean(equals.type(torch.FloatTensor)).item()
            val_acc_tracker.append(epoch_acc)
            accuracy += epoch_acc
            
            # Print the progress of our evaluation
            counter += 1
            val_counter.append(counter)
            print("Evaluation Progress ({}/{})".format(counter, len(val_loader)), "Loss: {:.6f}".format(epoch_loss), "Accuracy: {:.6f} %".format(epoch_acc*100))
    
    val_loss = val_loss/len(val_loader.dataset)
    print("Testing Step Average Loss: {:.6f}".format(val_loss))

    # Print out the information
    print('Average Accuracy: {} %'.format((accuracy/len(val_loader))*100))
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \n\n'.format(epoch, train_loss, val_loss))

    model.eval()
    
    # Process our image
def process_image(image_path):
    # Load Image
    img = Image.open(image_path)
    
    # Get the dimensions of the image
    width, height = img.size
    
    # Resize by keeping the aspect ratio, but changing the dimension so the shortest size is 255px
    img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    
    # Get the dimensions of the new image size
    width, height = img.size
    
    # Set the coordinates to do a center crop of 224 x 224
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    img = img.crop((left, top, right, bottom))
    
    # Turn image into numpy array
    img = np.array(img)
    
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    image = image.float()
    return image

# Using our model to predict the label
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for that class
    accuracy, prediciton = output.topk(1, dim=1)

    return accuracy.item(), prediciton.item()

# Show Image
def show_image(image, prediciton):
    if str(device) == 'cuda':
        # Convert to CPU tensor
        image = image.cpu()

    # Convert image to numpy
    image = image.numpy()
    
    # Un-normalize the image
    image[0] = image[0] * 0.226 + 0.445
    
    # Remove Batch Dimension (Shift first to last)
    image = np.transpose(image[0], (1, 2, 0))

    # Print the image
    plt.figure()
    plt.title("Prediciton: {}".format(prediciton))
    plt.imshow(image)
    plt.show()

print("train_counter", train_counter)
print("train_loss_tracker: ", train_loss_tracker)
print("val_counter: ", val_counter)
print("val_loss_tracker: ", val_loss_tracker)

# Loss Curve
plt.scatter(train_counter, train_loss_tracker, color='lime')
plt.scatter(val_counter, val_loss_tracker, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training samples')
plt.ylabel('loss')
plt.show()

# Process Image
img_pred = []
img_pred0 = 'data/test/0/HandGestures-0-2.jpg'
img_pred1 = 'data/test/1/HandGestures-1-2.jpg'
img_pred2 = 'data/test/2/HandGestures-2-4.jpg'
img_pred3 = 'data/test/3/HandGestures-3-2.jpg'
img_pred4 = 'data/test/4/HandGestures-4-3.jpg'
img_pred5 = 'data/test/5/HandGestures-5-1.jpg'

img_pred.append(img_pred0)
img_pred.append(img_pred1)
img_pred.append(img_pred2)
img_pred.append(img_pred3)
img_pred.append(img_pred4)
img_pred.append(img_pred5)

y_pred = []
pred_acc = []
for i in range(0, len(img_pred)):
    if str(device) == 'cuda':
        image = process_image(img_pred[i]).to(device)
    else:
        image = process_image(img_pred[i])

    # Give image to model to predict output
    accuracy, prediction = predict(image, model)
    accuracy = accuracy*100
    pred_acc.append(accuracy)
    y_pred.append(prediction)

    # Show the image
    show_image(image, prediction)

print("Predicitons:")
model_pred = []
for i in range(0, len(y_pred)):   
    pred = "({:.2f}%) True Label {} : Predicted Label: {}".format(pred_acc[i], class_name[i], y_pred[i])
    model_pred.append(pred)
    print(pred)