import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Data Preparation (Building on previous implementation)
def add_symmetric_noise(dataset, noise_rate=0.5):
    targets = torch.tensor(dataset.targets)
    num_classes = len(dataset.classes)
    original_targets = targets.clone()
    
    for cls in range(num_classes):
        cls_idx = (original_targets == cls).nonzero().view(-1)
        n_noisy = int(noise_rate * len(cls_idx))
        noisy_idx = cls_idx[torch.randperm(len(cls_idx))[:n_noisy]]
        targets[noisy_idx] = torch.randint(0, num_classes, (n_noisy,))
    
    dataset.targets = targets.tolist()
    return dataset

# Loss Functions
class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        norm_factor = torch.logsumexp(inputs, dim=1)  # Following paper's Eq(2)
        return (ce_loss / norm_factor).mean()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma) * ce_loss
        norm_factor = torch.logsumexp(inputs, dim=1)
        return (focal_loss / norm_factor).mean()

# Training Function
def train_model(trainloader, testloader, loss_fn, noise_rate, n_epochs=100):
    model = torchvision.models.resnet18(num_classes=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    metrics = {'train_loss': [], 'test_acc': []}
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        metrics['train_loss'].append(epoch_loss/len(trainloader))
        
        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
        
        metrics['test_acc'].append(correct/len(testloader.dataset))
        print(f"η={noise_rate} Epoch {epoch+1}: Loss={metrics['train_loss'][-1]:.4f}, Acc={metrics['test_acc'][-1]:.4f}")
    
    return metrics

# Experiment Setup
noise_rates = [0.2, 0.4, 0.6, 0.8]
loss_fns = {
    'CE': torch.nn.CrossEntropyLoss(),
    'NCE': NormalizedCrossEntropy(),
    'FL': torch.nn.CrossEntropyLoss(),  # Will apply focal scaling
    'NFL': NormalizedFocalLoss()
}

# Main Experiment
results = {}
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

clean_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
clean_testloader = DataLoader(clean_testset, batch_size=256, shuffle=False)

for noise_rate in noise_rates:
    # Prepare noisy dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainset = add_symmetric_noise(trainset, noise_rate)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    
    # Train with different loss functions
    for loss_name in ['CE', 'NCE', 'FL', 'NFL']:
        key = f"η={noise_rate}_{loss_name}"
        print(f"\nTraining {key}")
        
        if loss_name == 'FL':  # Implement Focal Loss
            def focal_loss(inputs, targets):
                ce = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce)
                return ((1 - pt)**2 * ce).mean()
            
            results[key] = train_model(trainloader, clean_testloader, focal_loss, noise_rate)
        else:
            results[key] = train_model(trainloader, clean_testloader, loss_fns[loss_name], noise_rate)

# Visualization
plt.figure(figsize=(12, 5))

# Accuracy vs Noise Rate
plt.subplot(1, 2, 1)
for loss_name in ['CE', 'NCE', 'FL', 'NFL']:
    accs = [results[f"η={nr}_{loss_name}"]['test_acc'][-1] for nr in noise_rates]
    plt.plot(noise_rates, accs, marker='o', label=loss_name)
plt.xlabel('Noise Rate (η)')
plt.ylabel('Test Accuracy')
plt.title('Final Accuracy vs Noise Rate')
plt.legend()

# Training Curves
plt.subplot(1, 2, 2)
for key in [f"η=0.6_{ln}" for ln in ['CE', 'NCE']]:
    plt.plot(results[key]['test_acc'], label=key.split('_')[1])
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy')
plt.title('Training Progress (η=0.6)')
plt.legend()

plt.tight_layout()
plt.savefig('coreml_results.png')
plt.show()
