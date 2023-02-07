import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

colors = [
  [128, 64, 128],
  [244, 35, 232],
  [70, 70, 70],
  [102, 102, 156],
  [190, 153, 153],
  [153, 153, 153],
  [250, 170, 30],
  [220, 220, 0],
  [107, 142, 35],
  [152, 251, 152],
  [0, 130, 180],
  [220, 20, 60],
  [255, 0, 0],
  [0, 0, 142],
  [0, 0, 70],
  [0, 60, 100],
  [0, 80, 100],
  [0, 0, 230],
  [119, 11, 32]
]

def decode_segmap(image):
  #convert gray scale to color
  image = image.numpy()
  r = image.copy()
  g = image.copy()
  b = image.copy()

  label_colours = dict(zip(range(19), colors))
  for l in range(0, 19):
    r[image == l] = label_colours[l][0]
    g[image == l] = label_colours[l][1]
    b[image == l] = label_colours[l][2]

  rgb = np.zeros((image.shape[0], image.shape[1], 3))
  rgb[:, :, 0] = r / 255.0
  rgb[:, :, 1] = g / 255.0
  rgb[:, :, 2] = b / 255.0
  return rgb

def display_prediction(net, dataset, random=False):    
    if random == True:
      dl = DataLoader(dataset, batch_size=1, shuffle=True)
    else:
      dl = DataLoader(dataset, batch_size=1, shuffle=False)
    for (i, l) in dl:
      break

    fig = plt.figure(figsize=(20, 10))

    fig.add_subplot(1, 3, 1)
    plt.imshow(i[0].permute(1, 2, 0))
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    l_ = l[0].squeeze()
    l_ = decode_segmap(l_)
    plt.imshow(l_)
    plt.axis('off')

    net.eval()
    with torch.no_grad():
      i = i.to(DEVICE)
      out = net(i)
      out = torch.softmax(out, dim=1)
      out = torch.argmax(out, dim=1)
      out = out.cpu().squeeze()

      fig.add_subplot(1, 3, 3)
      out = decode_segmap(out)
      plt.imshow(out)
      plt.axis('off')
    plt.show()