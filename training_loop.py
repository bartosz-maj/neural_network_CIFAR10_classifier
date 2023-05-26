# Lists are defined to store accuracies, loss, epochs and the learning rate over time for evaluation purposes
training_accuracies = []
testing_accuracies = []
loss_over_time = []
epochs = []
learning_rates = []
# Number of epochs is defined
epoch_num = 80
# Loops through the dataset for the predefined number of epochs
for epoch in range(epoch_num):  

    running_loss = 0.0
    epoch_loss = []
    for i, data in enumerate(trainloader, 0):
        # Inputs are extracted from the train set dataloader. 
        inputs, labels = data
        # Data is then sent to the gpu.
        inputs, labels = inputs.to(dev), labels.to(dev)

        # Gradients are set to zero to ensure gradients from the previous loop are not left over. 
        optimizer.zero_grad()

        # The network produces an input
        outputs = simple_net(inputs)
        # A loss value is calculated
        loss = criterion(outputs, labels)
        # Gradients for the loss are generated 
        loss.backward()
        # Gradients are used to update optimizer
        optimizer.step()
          
        epoch_loss.append(loss.item())
        # Loss is kept track of and printed throughout each epoch. 
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
            
    #keeps track of epoch loss and computes an average
    loss_over_time.append(sum(epoch_loss)/len(epoch_loss))
    # Correct prediction variables are created to allow further evaluation. 
    train_correct = 0
    train_total = 0
    
    # torch.no_grad() is set to ensure the gradients are not updated whilst the model is being evaluated 
    # on the training data
    with torch.no_grad():
        for train_data in trainloader:
            # Data is loaded
            train_images, train_labels = train_data
                # Data is sent to gpu
            train_images, train_labels = train_images.to(dev), train_labels.to(dev)
                # Data is put through model
            train_outputs = simple_net(train_images)
                # Predictions are extracted
            _, predicted = torch.max(train_outputs.data, 1)
                # Total predictions and correct predictions are updated 
                # to calculate accuracy metric. 
            train_total += train_labels.size(0)
            train_correct += (predicted == train_labels).sum().item()
    
    # Training accuracies are added to their list and printed. 
    training_accuracies.append(100 * train_correct // train_total)
    print(f'Training accuracy of the network on the 10000 test images: {100 * train_correct // train_total} %')
    
    # The process of assessing the model on the validation set is the same as assessing it on the training set
    # with the only difference being that the simple_net.eval() is put the model into its evlauation mode. 
    test_correct = 0
    test_total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    simple_net.eval()
    with torch.no_grad():
        for test_data in testloader:
            test_images, test_labels = test_data
            test_images, test_labels = test_images.to(dev), test_labels.to(dev)
            # calculate outputs by running images through the network
            test_outputs = simple_net(test_images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(test_outputs.data, 1)
            test_total += test_labels.size(0)
            test_correct += (predicted == test_labels).sum().item()
    testing_accuracies.append(100 * test_correct // test_total)
    print(f'Test accuracy of the network on the 10000 test images: {100 * test_correct // test_total} %')
    simple_net.train()
    
    
    epochs.append(epoch)
    
    # The scheduler is given the test accuracy to check whether it needs to update the learning rate. 
    scheduler.step(100 * test_correct // test_total)

    for param_group in optimizer.param_groups:
         learning_rates.append(param_group['lr'])

print('Finished Training')
