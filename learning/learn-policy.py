import torch as th
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import NeuralNet, PolicyModel
from dataset import Dataset
from consts import GAME_LENGTH

EPOCHS = 5


dataset = Dataset('experience-tiger-policy.txt', featuresCols=range(0,50), targetCol=[50, 51])
dataloader = th.utils.data.DataLoader(dataset, batch_size=GAME_LENGTH)


device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')

# LRs = [0.001, 0.01, 0.0025]
LRs = [0.005]
for LR in LRs:
    model = PolicyModel().to(device)
    # model = th.load('tigerModel-learn.pt')
    optimizer = th.optim.Adam(model.parameters(), lr=LR)
    loss_func = th.nn.CrossEntropyLoss()

    avg_loss = []
    for _ in tqdm(range(EPOCHS)):
        total_loss = 0
        for data, target in tqdm(dataloader, leave=False):
            optimizer.zero_grad()

            data = data.float().to(device)
            pred = model(data)

            # print('the target', target)
            target, reward = th.split(target, [1, 1], dim=1)
            target = target.flatten()

            loss = loss_func(pred, target)
            loss = th.mean(loss * reward)
            loss.backward()

            total_loss += loss.item()

            optimizer.step()
        
        avg_loss.append(total_loss / (len(dataloader) * GAME_LENGTH))

    plt.plot(range(EPOCHS), avg_loss)

    th.save(model, 'model-tiger-policy-learn.pt')

plt.legend(['LR: %f'% LR for LR in LRs])
plt.show()