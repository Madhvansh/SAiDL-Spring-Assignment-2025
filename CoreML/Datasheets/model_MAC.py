import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cifar10_noisy import add_symmetric_noise
from torchvision.datasets import CIFAR10

# 1. Enhanced Normalized Loss Functions ========================================
class NormalizedCrossEntropy(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        ce = -torch.log(probs[torch.arange(logits.size(0)), targets] + self.eps)
        norm = -torch.sum(probs * torch.log(probs + self.eps), dim=1)
        return torch.mean(ce / (norm + self.eps))

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

# 2. Device-Agnostic Model Architecture =======================================
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

# 3. Robust Training Pipeline =================================================
def train_model(eta, loss_fn, num_epochs=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preparation with controlled η
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Add noise with specified η
    noisy_train, _, _, _ = add_symmetric_noise(train_set, eta=eta, seed=42)
    
    train_loader = DataLoader(noisy_train, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # Model setup
    model = ResNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training metrics
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
        
        # Evaluation
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
        
        # Update metrics
        avg_loss = epoch_loss / len(train_loader.dataset)
        test_acc = 100 * correct / total
        metrics['train_loss'].append(avg_loss)
        metrics['test_acc'].append(test_acc)
        metrics['best_acc'] = max(metrics['best_acc'], test_acc)
        
        print(f'Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {test_acc:.2f}%')
        scheduler.step()
    
    return metrics

# 4. Comprehensive Experiment Runner ==========================================
def run_experiments():
    torch.manual_seed(42)
    eta_values = [0.2, 0.4, 0.6, 0.8]
    loss_configs = {
        'CE': nn.CrossEntropyLoss(),
        'FL': NormalizedFocalLoss(gamma=0),  # Vanilla FL
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
    
    # Visualization
    for eta in eta_values:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for loss_name in loss_configs:
            plt.plot(results[eta][loss_name]['loss_curve'], label=loss_name)
        plt.title(f'Training Loss (η={eta})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for loss_name in loss_configs:
            plt.plot(results[eta][loss_name]['acc_curve'], label=loss_name)
        plt.title(f'Test Accuracy (η={eta})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Final comparison table
    print("\nFinal Comparison:")
    print(f"{'Loss':<6} {'η=0.2':<8} {'η=0.4':<8} {'η=0.6':<8} {'η=0.8':<8}")
    for loss_name in loss_configs:
        acc_values = [f"{results[eta][loss_name]['best_acc']:.2f}%" for eta in eta_values]
        print(f"{loss_name:<6} {' '.join(acc_values)}")

if __name__ == "__main__":
    run_experiments()
