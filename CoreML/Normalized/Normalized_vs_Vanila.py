import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
logging.getLogger("torchvision.datasets").setLevel(logging.WARNING)

def add_symmetric_noise(dataset, eta = None, seed=42):
    # Generate random η if not provided
    if eta is None:
        eta = torch.empty(1).uniform_(0.2, 0.8).item()
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    targets = torch.tensor(dataset.targets)
    num_classes = len(dataset.classes)
    noise_matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    original_targets = targets.clone()

    for orig_class in range(num_classes):
        class_mask = original_targets == orig_class
        class_indices = class_mask.nonzero(as_tuple=True)[0]
        n_samples = class_indices.size(0)
        n_noisy = int(eta * n_samples)
        
        if n_noisy > 0:
            # Randomly select samples to corrupt using PyTorch
            perm = torch.randperm(n_samples)[:n_noisy]
            noisy_idx = class_indices[perm]
            
            # Generate new labels using PyTorch
            possible_classes = torch.arange(num_classes, device=targets.device)
            possible_classes = possible_classes[possible_classes != orig_class]
            new_labels = possible_classes[torch.randint(0, len(possible_classes), (n_noisy,))]
            
            # Update targets and noise matrix
            targets[noisy_idx] = new_labels
            unique_new, counts = torch.unique(new_labels, return_counts=True)
            noise_matrix[orig_class].index_add_(0, unique_new, counts)
            noise_matrix[orig_class, orig_class] += n_samples - n_noisy

    # Convert back to list for dataset compatibility
    dataset.targets = targets.tolist()
    
    return dataset, noise_matrix, original_targets, eta

# Load CIFAR-10 dataset
train_dataset = CIFAR10(
    root='./data', 
    train=True, 
    download=True,
    transform=transforms.ToTensor()
)

# Then pass it to the noise function
noisy_dataset, noise_matrix, original_targets, eta = add_symmetric_noise(train_dataset)
noisy_targets = torch.tensor(noisy_dataset.targets)

# Print statistics
total_samples = original_targets.size(0)
corrupted_count = (original_targets != noisy_targets).sum().item()
print(f"\n Generated η: {eta:.2f}")
print(f"Total samples: {total_samples}")
print(f"Corrupted samples: {corrupted_count} ({corrupted_count/total_samples:.2%})\n")


class NormalizedCrossEntropy(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        ce = -torch.log(probs[torch.arange(logits.size(0)), targets] + 1e-8)
        norm = -torch.sum(torch.log(probs + 1e-8), dim=1)  # Sum(CE)
        return torch.mean(ce / norm)


class NormalizedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        focal = (1 - probs[torch.arange(logits.size(0)), targets])**self.gamma * \
               -torch.log(probs[torch.arange(logits.size(0)), targets] + self.eps)
        sum_fl = torch.sum((1 - probs)**self.gamma * -torch.log(probs + self.eps), dim=1)
        return torch.mean(focal / (sum_fl + self.eps))
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

def train_model(eta, loss_fn, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    noisy_train, _, _, _ = add_symmetric_noise(train_set, eta=eta, seed=42)
    
    train_loader = DataLoader(noisy_train, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = ResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    metrics = {
        'train_loss': [],
        'test_acc': [],
        'best_acc': 0.0
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        avg_loss = epoch_loss / len(train_loader.dataset)
        test_acc = 100 * correct / total
        metrics['train_loss'].append(avg_loss)
        metrics['test_acc'].append(test_acc)
        metrics['best_acc'] = max(metrics['best_acc'], test_acc)
        
        print(f'Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {test_acc:.2f}%')
        scheduler.step()
    
    return metrics

def run_experiments():
    torch.manual_seed(42)
    eta_values = [0.2, 0.4, 0.6, 0.8]
    loss_configs = {
        'CE': nn.CrossEntropyLoss(),
        'FL': FocalLoss(gamma=2),
        'NCE': NormalizedCrossEntropy(),
        'NFL': NormalizedFocalLoss(gamma=2)
    }
    
    results = {eta: {} for eta in eta_values}
    
    for eta in eta_values:
        print(f"\n=== Training at η={eta} ===")
        
        for loss_name, loss_fn in loss_configs.items():
            print(f"\nTraining {loss_name}...")
            metrics = train_model(eta, loss_fn)
            results[eta][loss_name] = {
                'best_acc': metrics['best_acc'],
                'loss_curve': metrics['train_loss'],
                'acc_curve': metrics['test_acc']
            }
    
    print("\nFinal Comparison:")
    print(f"{'Loss':<6} {'η=0.2':<8} {'η=0.4':<8} {'η=0.6':<8} {'η=0.8':<8}")
    for loss_name in loss_configs:
        acc_values = [f"{results[eta][loss_name]['best_acc']:.2f}%" for eta in eta_values]
        print(f"{loss_name:<6} {' '.join(acc_values)}")

if __name__ == "__main__":
    run_experiments()