import torch as th
import csv
from tqdm import tqdm
from matplotlib import pyplot as plt

from model import NeuralNet
from dataset import Dataset

EPOCHS = 5

# with open('experience.txt') as file:
#     reader = csv.reader(file, delimiter=',')
#     experiences = list(reader)

dataset = Dataset('experience-big.txt')
dataloader = th.utils.data.DataLoader(dataset, batch_size=40)


device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')


# LRs = [0.001, 0.01, 0.0025]
LRs = [0.0025]
for LR in LRs:
    # model = NeuralNet().to(device)
    model = th.load('tigerModel-learn.pt')
    optimizer = th.optim.Adam(model.parameters(), lr=LR)
    loss_func = th.nn.MSELoss()

    avg_loss = []
    for _ in tqdm(range(EPOCHS)):
        total_loss = 0
        for inp, target in tqdm(dataloader, leave=False):
            # print(inp)
            # print(type(th.tensor(inp)))
            # break
            optimizer.zero_grad()

            inp = inp.float().to(device)
            pred = model.layers(inp)

            target = target.float().reshape(-1, 1)
            loss = loss_func(pred, target).to(device)
            loss.backward()

            total_loss += loss.item()

            optimizer.step()
        
        avg_loss.append(total_loss / len(experiences))

    plt.plot(range(EPOCHS), avg_loss)

    th.save(model, 'tigerModel-learn-big.pt')

plt.legend(['LR: %f'% LR for LR in LRs])
plt.show()