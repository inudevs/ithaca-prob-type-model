import torch.optim as optim
import torch
import numpy as np
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from model import Net, transform

classes = ['english', 'math']

batch_size = 32
test_size = 0.3
valid_size = 0.1

data = datasets.ImageFolder('data', transform=transform)

# For test
num_data = len(data)
print(num_data)
indices_data = list(range(num_data))
np.random.shuffle(indices_data)
split_tt = int(np.floor(test_size * num_data))
print(split_tt)
train_idx, test_idx = indices_data[split_tt:], indices_data[:split_tt]

# For Valid
num_train = len(train_idx)
indices_train = list(range(num_train))
np.random.shuffle(indices_train)
split_tv = int(np.floor(valid_size * num_train))
train_new_idx, valid_idx = indices_train[split_tv:], indices_train[:split_tv]

train_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_new_idx),
    num_workers=1)
valid_loader = torch.utils.data.DataLoader(
    data,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(valid_idx),
    num_workers=1)
test_loader = torch.utils.data.DataLoader(
    data,
    sampler=SubsetRandomSampler(test_idx),
    batch_size=batch_size,
    num_workers=1)

print(
    len(test_loader) *
    batch_size +
    len(valid_loader) *
    batch_size +
    len(train_loader) *
    batch_size)

for batch in valid_loader:
    print(batch[0].size())


model = Net()
print(model)


# loss function
criterion = torch.nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

n_epochs = 10
valid_loss_min = np.Inf

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0

    # train
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    # validation
    model.eval()
    for data, target in valid_loader:
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = valid_loss / len(valid_loader.dataset)

    print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, n_epochs, train_loss, valid_loss))

    # save model
    if valid_loss <= valid_loss_min:
        print(
            'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss


# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

model.eval()
i = 1
len(test_loader)
for data, target in test_loader:
    i += 1
    if len(target) != batch_size:
        continue
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())

    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss / len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(2):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print(
            'Test Accuracy of %5s: N/A (no training examples)' %
            (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
