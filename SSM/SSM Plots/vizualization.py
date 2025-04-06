import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import re

# 1. Parse training logs from notebook output
log_data = """
Epoch 1/200 | Loss: 2.1616 | Acc: 30.73%
Epoch 2/200 | Loss: 2.0418 | Acc: 36.84%
Epoch 3/200 | Loss: 1.9760 | Acc: 40.72%
Epoch 4/200 | Loss: 1.9081 | Acc: 45.08%
Epoch 5/200 | Loss: 1.8538 | Acc: 48.76%
Epoch 6/200 | Loss: 1.8152 | Acc: 49.32%
Epoch 7/200 | Loss: 1.7827 | Acc: 51.71%
Epoch 8/200 | Loss: 1.7585 | Acc: 52.02%
Epoch 9/200 | Loss: 1.7341 | Acc: 56.01%
Epoch 10/200 | Loss: 1.7085 | Acc: 53.82%
Epoch 11/200 | Loss: 1.6834 | Acc: 56.49%
Epoch 12/200 | Loss: 1.6629 | Acc: 57.05%
Epoch 13/200 | Loss: 1.6423 | Acc: 60.72%
Epoch 14/200 | Loss: 1.6252 | Acc: 60.62%
Epoch 15/200 | Loss: 1.6078 | Acc: 61.41%
Epoch 16/200 | Loss: 1.5895 | Acc: 63.20%
Epoch 17/200 | Loss: 1.5740 | Acc: 62.68%
Epoch 18/200 | Loss: 1.5558 | Acc: 64.01%
Epoch 19/200 | Loss: 1.5477 | Acc: 64.48%
Epoch 20/200 | Loss: 1.5354 | Acc: 65.59%
Epoch 21/200 | Loss: 1.5259 | Acc: 64.59%
Epoch 22/200 | Loss: 1.5122 | Acc: 65.72%
Epoch 23/200 | Loss: 1.5006 | Acc: 66.14%
Epoch 24/200 | Loss: 1.4890 | Acc: 65.75%
Epoch 25/200 | Loss: 1.4830 | Acc: 68.60%
Epoch 26/200 | Loss: 1.4699 | Acc: 68.19%
Epoch 27/200 | Loss: 1.4631 | Acc: 69.09%
Epoch 28/200 | Loss: 1.4526 | Acc: 67.93%
Epoch 29/200 | Loss: 1.4445 | Acc: 69.72%
Epoch 30/200 | Loss: 1.4346 | Acc: 69.41%
Epoch 31/200 | Loss: 1.4277 | Acc: 68.84%
Epoch 32/200 | Loss: 1.4206 | Acc: 70.38%
Epoch 33/200 | Loss: 1.4124 | Acc: 71.07%
Epoch 34/200 | Loss: 1.4036 | Acc: 70.56%
Epoch 35/200 | Loss: 1.3989 | Acc: 72.17%
Epoch 36/200 | Loss: 1.3925 | Acc: 71.51%
Epoch 37/200 | Loss: 1.3831 | Acc: 72.47%
Epoch 38/200 | Loss: 1.3746 | Acc: 73.44%
Epoch 39/200 | Loss: 1.3642 | Acc: 72.71%
Epoch 40/200 | Loss: 1.3584 | Acc: 72.99%
Epoch 41/200 | Loss: 1.3543 | Acc: 74.22%
Epoch 42/200 | Loss: 1.3474 | Acc: 73.37%
Epoch 43/200 | Loss: 1.3449 | Acc: 73.55%
Epoch 44/200 | Loss: 1.3370 | Acc: 73.13%
Epoch 45/200 | Loss: 1.3269 | Acc: 74.97%
Epoch 46/200 | Loss: 1.3225 | Acc: 73.92%
Epoch 47/200 | Loss: 1.3267 | Acc: 75.11%
Epoch 48/200 | Loss: 1.3102 | Acc: 75.29%
Epoch 49/200 | Loss: 1.3130 | Acc: 74.98%
Epoch 50/200 | Loss: 1.3033 | Acc: 75.61%
Epoch 51/200 | Loss: 1.3068 | Acc: 75.86%
Epoch 52/200 | Loss: 1.3010 | Acc: 76.21%
Epoch 53/200 | Loss: 1.2915 | Acc: 76.12%
Epoch 54/200 | Loss: 1.2874 | Acc: 76.43%
Epoch 55/200 | Loss: 1.2775 | Acc: 76.07%
Epoch 56/200 | Loss: 1.2766 | Acc: 77.09%
Epoch 57/200 | Loss: 1.2719 | Acc: 76.63%
Epoch 58/200 | Loss: 1.2713 | Acc: 76.90%
Epoch 59/200 | Loss: 1.2709 | Acc: 76.15%
Epoch 60/200 | Loss: 1.2665 | Acc: 77.03%
Epoch 61/200 | Loss: 1.2547 | Acc: 77.57%
Epoch 62/200 | Loss: 1.2494 | Acc: 77.06%
Epoch 63/200 | Loss: 1.2493 | Acc: 77.90%
Epoch 64/200 | Loss: 1.2470 | Acc: 77.42%
Epoch 65/200 | Loss: 1.2458 | Acc: 77.85%
Epoch 66/200 | Loss: 1.2387 | Acc: 78.87%
Epoch 67/200 | Loss: 1.2329 | Acc: 78.79%
Epoch 68/200 | Loss: 1.2282 | Acc: 79.15%
Epoch 69/200 | Loss: 1.2225 | Acc: 79.36%
Epoch 70/200 | Loss: 1.2250 | Acc: 77.78%
Epoch 71/200 | Loss: 1.2227 | Acc: 78.56%
Epoch 72/200 | Loss: 1.2188 | Acc: 79.38%
Epoch 73/200 | Loss: 1.2125 | Acc: 79.54%
Epoch 74/200 | Loss: 1.2136 | Acc: 80.20%
Epoch 75/200 | Loss: 1.2016 | Acc: 79.44%
Epoch 76/200 | Loss: 1.2041 | Acc: 80.12%
Epoch 77/200 | Loss: 1.1992 | Acc: 79.51%
Epoch 78/200 | Loss: 1.1926 | Acc: 79.97%
Epoch 79/200 | Loss: 1.1942 | Acc: 80.17%
Epoch 80/200 | Loss: 1.1841 | Acc: 80.10%
Epoch 81/200 | Loss: 1.1842 | Acc: 80.44%
Epoch 82/200 | Loss: 1.1855 | Acc: 80.37%
Epoch 83/200 | Loss: 1.1789 | Acc: 79.81%
Epoch 84/200 | Loss: 1.1788 | Acc: 80.57%
Epoch 85/200 | Loss: 1.1748 | Acc: 80.11%
Epoch 86/200 | Loss: 1.1743 | Acc: 81.02%
Epoch 87/200 | Loss: 1.1688 | Acc: 81.27%
Epoch 88/200 | Loss: 1.1610 | Acc: 81.20%
Epoch 89/200 | Loss: 1.1608 | Acc: 80.85%
Epoch 90/200 | Loss: 1.1591 | Acc: 81.35%
Epoch 91/200 | Loss: 1.1551 | Acc: 81.77%
Epoch 92/200 | Loss: 1.1533 | Acc: 81.32%
Epoch 93/200 | Loss: 1.1513 | Acc: 81.89%
Epoch 94/200 | Loss: 1.1477 | Acc: 81.62%
Epoch 95/200 | Loss: 1.1439 | Acc: 82.53%
Epoch 96/200 | Loss: 1.1376 | Acc: 81.79%
Epoch 97/200 | Loss: 1.1387 | Acc: 82.23%
Epoch 98/200 | Loss: 1.1376 | Acc: 81.60%
Epoch 99/200 | Loss: 1.1271 | Acc: 82.01%
Epoch 100/200 | Loss: 1.1224 | Acc: 81.70%
Epoch 101/200 | Loss: 1.1231 | Acc: 82.32%
Epoch 102/200 | Loss: 1.1228 | Acc: 82.30%
Epoch 103/200 | Loss: 1.1187 | Acc: 83.01%
Epoch 104/200 | Loss: 1.1220 | Acc: 82.59%
Epoch 105/200 | Loss: 1.1113 | Acc: 82.77%
Epoch 106/200 | Loss: 1.1103 | Acc: 82.61%
Epoch 107/200 | Loss: 1.1112 | Acc: 83.17%
Epoch 108/200 | Loss: 1.1026 | Acc: 83.06%
Epoch 109/200 | Loss: 1.1013 | Acc: 83.09%
Epoch 110/200 | Loss: 1.0965 | Acc: 83.39%
Epoch 111/200 | Loss: 1.0947 | Acc: 83.01%
Epoch 112/200 | Loss: 1.0885 | Acc: 83.32%
Epoch 113/200 | Loss: 1.0899 | Acc: 83.37%
Epoch 114/200 | Loss: 1.0861 | Acc: 83.72%
Epoch 115/200 | Loss: 1.0871 | Acc: 84.06%
Epoch 116/200 | Loss: 1.0806 | Acc: 83.39%
Epoch 117/200 | Loss: 1.0809 | Acc: 83.88%
Epoch 118/200 | Loss: 1.0770 | Acc: 83.92%
Epoch 119/200 | Loss: 1.0773 | Acc: 83.59%
Epoch 120/200 | Loss: 1.0697 | Acc: 83.87%
Epoch 121/200 | Loss: 1.0693 | Acc: 83.57%
Epoch 122/200 | Loss: 1.0649 | Acc: 84.14%
Epoch 123/200 | Loss: 1.0593 | Acc: 84.40%
Epoch 124/200 | Loss: 1.0562 | Acc: 84.58%
Epoch 125/200 | Loss: 1.0545 | Acc: 84.18%
Epoch 126/200 | Loss: 1.0493 | Acc: 84.44%
Epoch 127/200 | Loss: 1.0521 | Acc: 84.40%
Epoch 128/200 | Loss: 1.0481 | Acc: 84.49%
Epoch 129/200 | Loss: 1.0447 | Acc: 84.30%
Epoch 130/200 | Loss: 1.0422 | Acc: 84.56%
Epoch 131/200 | Loss: 1.0419 | Acc: 85.03%
Epoch 132/200 | Loss: 1.0358 | Acc: 84.60%
Epoch 133/200 | Loss: 1.0413 | Acc: 84.90%
Epoch 134/200 | Loss: 1.0323 | Acc: 85.03%
Epoch 135/200 | Loss: 1.0248 | Acc: 85.13%
Epoch 136/200 | Loss: 1.0266 | Acc: 85.35%
Epoch 137/200 | Loss: 1.0265 | Acc: 85.05%
Epoch 138/200 | Loss: 1.0219 | Acc: 84.86%
Epoch 139/200 | Loss: 1.0157 | Acc: 84.93%
Epoch 140/200 | Loss: 1.0111 | Acc: 85.10%
Epoch 141/200 | Loss: 1.0168 | Acc: 85.24%
Epoch 142/200 | Loss: 1.0130 | Acc: 84.83%
Epoch 143/200 | Loss: 1.0090 | Acc: 85.35%
Epoch 144/200 | Loss: 1.0031 | Acc: 85.34%
Epoch 145/200 | Loss: 1.0049 | Acc: 85.00%
Epoch 146/200 | Loss: 0.9987 | Acc: 85.44%
Epoch 147/200 | Loss: 0.9982 | Acc: 85.73%
Epoch 148/200 | Loss: 0.9943 | Acc: 85.82%
Epoch 149/200 | Loss: 0.9996 | Acc: 85.70%
Epoch 150/200 | Loss: 0.9875 | Acc: 85.68%
Epoch 151/200 | Loss: 0.9839 | Acc: 85.81%
Epoch 152/200 | Loss: 0.9843 | Acc: 85.92%
Epoch 153/200 | Loss: 0.9824 | Acc: 86.06%
Epoch 154/200 | Loss: 0.9813 | Acc: 85.05%
Epoch 155/200 | Loss: 0.9771 | Acc: 85.59%
Epoch 156/200 | Loss: 0.9759 | Acc: 85.79%
Epoch 157/200 | Loss: 0.9835 | Acc: 86.20%
Epoch 158/200 | Loss: 0.9699 | Acc: 86.02%
Epoch 159/200 | Loss: 0.9723 | Acc: 86.11%
Epoch 160/200 | Loss: 0.9738 | Acc: 86.00%
Epoch 161/200 | Loss: 0.9661 | Acc: 86.41%
Epoch 162/200 | Loss: 0.9687 | Acc: 86.24%
Epoch 163/200 | Loss: 0.9658 | Acc: 86.09%
Epoch 164/200 | Loss: 0.9642 | Acc: 86.25%
Epoch 165/200 | Loss: 0.9607 | Acc: 86.45%
Epoch 166/200 | Loss: 0.9529 | Acc: 86.27%
Epoch 167/200 | Loss: 0.9596 | Acc: 86.49%
Epoch 168/200 | Loss: 0.9535 | Acc: 86.43%
Epoch 169/200 | Loss: 0.9566 | Acc: 86.69%
Epoch 170/200 | Loss: 0.9505 | Acc: 86.43%
Epoch 171/200 | Loss: 0.9505 | Acc: 86.62%
Epoch 172/200 | Loss: 0.9451 | Acc: 86.61%
Epoch 173/200 | Loss: 0.9494 | Acc: 86.57%
Epoch 174/200 | Loss: 0.9483 | Acc: 86.63%
Epoch 175/200 | Loss: 0.9452 | Acc: 86.65%
Epoch 176/200 | Loss: 0.9427 | Acc: 86.59%
Epoch 177/200 | Loss: 0.9451 | Acc: 86.62%
Epoch 178/200 | Loss: 0.9436 | Acc: 86.83%
Epoch 179/200 | Loss: 0.9442 | Acc: 86.70%
Epoch 180/200 | Loss: 0.9385 | Acc: 86.64%
Epoch 181/200 | Loss: 0.9386 | Acc: 86.69%
Epoch 182/200 | Loss: 0.9431 | Acc: 86.54%
Epoch 183/200 | Loss: 0.9398 | Acc: 86.57%
Epoch 184/200 | Loss: 0.9384 | Acc: 86.72%
Epoch 185/200 | Loss: 0.9347 | Acc: 86.62%
Epoch 186/200 | Loss: 0.9341 | Acc: 86.63%
Epoch 187/200 | Loss: 0.9349 | Acc: 86.65%
Epoch 188/200 | Loss: 0.9333 | Acc: 86.63%
Epoch 189/200 | Loss: 0.9327 | Acc: 86.65%
Epoch 190/200 | Loss: 0.9346 | Acc: 86.71%
Epoch 191/200 | Loss: 0.9350 | Acc: 86.67%
Epoch 192/200 | Loss: 0.9351 | Acc: 86.76%
Epoch 193/200 | Loss: 0.9305 | Acc: 86.74%
Epoch 194/200 | Loss: 0.9341 | Acc: 86.73%
Epoch 195/200 | Loss: 0.9309 | Acc: 86.70%
Epoch 196/200 | Loss: 0.9282 | Acc: 86.69%
Epoch 197/200 | Loss: 0.9336 | Acc: 86.61%
Epoch 198/200 | Loss: 0.9290 | Acc: 86.63%
Epoch 199/200 | Loss: 0.9335 | Acc: 86.63%
Epoch 200/200 | Loss: 0.9280 | Acc: 86.63%
"""

def parse_logs(log_str):
    epochs, losses, accuracies = [], [], []
    for line in log_str.split('\n'):
        if 'Epoch' in line:
            parts = re.findall(r'\d+\.\d+', line)
            if len(parts) >= 2:
                losses.append(float(parts[0]))
                accuracies.append(float(parts[1]))
            epoch = int(re.search(r'Epoch (\d+)/', line).group(1))
            epochs.append(epoch)
    return epochs, losses, accuracies

epochs, losses, accuracies = parse_logs(log_data)

# 2. Configure plot style
plt.style.use('seaborn-v0_8')
colors = plt.cm.viridis(np.linspace(0, 1, 3))
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 300
})

# 3. Training Loss Curve
plt.figure()
plt.plot(epochs, losses, color=colors[0], lw=2, alpha=0.8)
plt.title('Training Loss Progression')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.grid(True, alpha=0.3)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('training_loss.pdf', bbox_inches='tight')

# 4. Test Accuracy Curve
plt.figure()
plt.plot(epochs, accuracies, color=colors[1], lw=2, alpha=0.8)
plt.title('Test Accuracy Progression')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.ylim(0, 100)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.savefig('test_accuracy.pdf', bbox_inches='tight')

# 5. Dual Axis Training Dynamics
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(epochs, losses, color=colors[0], lw=2, label='Loss')
ax2.plot(epochs, accuracies, color=colors[1], lw=2, label='Accuracy')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=colors[0])
ax2.set_ylabel('Accuracy (%)', color=colors[1])
ax1.tick_params(axis='y', labelcolor=colors[0])
ax2.tick_params(axis='y', labelcolor=colors[1])
plt.title('Training Dynamics: Loss vs Accuracy')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
plt.savefig('dual_axis.pdf', bbox_inches='tight')

# 6. Phase Analysis Plot
def calculate_moving_avg(data, window=10):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure()
windows = [5, 10, 20]
for w in windows:
    plt.plot(epochs[w-1:], calculate_moving_avg(losses, w), 
             lw=1.5, label=f'{w}-epoch MA')

plt.title('Loss Trend Analysis with Moving Averages')
plt.xlabel('Epochs')
plt.ylabel('Smoothed Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('moving_avg.pdf', bbox_inches='tight')

# 7. Confusion Matrix (Mockup - Needs True Labels)
# For actual implementation, collect predictions during evaluation
classes = ['airplane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Mock confusion matrix data
cm = np.random.randn(10, 10)
np.fill_diagonal(cm, np.abs(np.diag(cm)) + 5)  # Emphasize diagonal

plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(np.arange(10), classes, rotation=45)
plt.yticks(np.arange(10), classes)
plt.title('Confusion Matrix (Mock Data)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.pdf')

# 8. Learning Rate Schedule Visualization 
# (Requires LR tracking during training)

# Mock data based on OneCycle policy
def mock_lr_schedule(epochs):
    lr = np.zeros_like(epochs)
    peak = 150
    for i, e in enumerate(epochs):
        if e < peak:
            lr[i] = 5e-4 * (e/peak)
        else:
            lr[i] = 5e-4 * (1 - (e-peak)/(len(epochs)-peak))
            
    return lr

lrs = mock_lr_schedule(epochs)

plt.figure()
plt.plot(epochs, lrs, color=colors[2], lw=2)
plt.title('Learning Rate Schedule')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.grid(True, alpha=0.3)
plt.savefig('lr_schedule.pdf', bbox_inches='tight')
