import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10
from cifar10_noisy import add_symmetric_noise  # Your existing noise function
# Normalized Loss Functions
class NormalizedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        norm_factor = torch.logsumexp(inputs, dim=1)
        return (ce_loss / norm_factor).mean()

class NormalizedFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma) * ce_loss
        norm_factor = torch.logsumexp(inputs, dim=1)
        return (focal_loss / norm_factor).mean()

# Training Module
class NoiseRobustTrainer:
    def __init__(self, train_loader, test_loader, device='cuda'):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        self.model = resnet18(num_classes=10).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, 
                                 momentum=0.9, weight_decay=5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200)
        
        self.loss_fns = {
            'CE': nn.CrossEntropyLoss(),
            'NCE': NormalizedCrossEntropy(),
            'FL': nn.CrossEntropyLoss(),  # Will be modified in train
            'NFL': NormalizedFocalLoss()
        }
        
    def train_epoch(self, loss_fn):
        self.model.train()
        epoch_loss = 0
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            if isinstance(loss_fn, str) and loss_fn == 'FL':
                ce = nn.functional.cross_entropy(outputs, targets, reduction='none')
                pt = torch.exp(-ce)
                loss = ((1 - pt)**2 * ce).mean()
            else:
                loss = loss_fn(outputs, targets)
                
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
        self.scheduler.step()
        return epoch_loss / len(self.train_loader)
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return correct / total
    
    def run_experiment(self, noise_rate, n_epochs=200):
        results = {}
        for loss_name in ['CE', 'NCE', 'FL', 'NFL']:
            print(f"Training with {loss_name} (η={noise_rate})")
            
            # Reset model for each experiment
            self.model.load_state_dict(resnet18(num_classes=10).state_dict())
            self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, 
                                     momentum=0.9, weight_decay=5e-4)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs)
            
            metrics = {'train_loss': [], 'test_acc': []}
            for epoch in range(n_epochs):
                train_loss = self.train_epoch(self.loss_fns[loss_name])
                test_acc = self.evaluate()
                
                metrics['train_loss'].append(train_loss)
                metrics['test_acc'].append(test_acc)
                
                if (epoch+1) % 20 == 0:
                    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={test_acc:.4f}")
            
            results[loss_name] = metrics
        return results

# Experiment Configuration
def run_experiments(noise_rates=[0.2, 0.4, 0.6, 0.8]):
    # Data transforms
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2470, 0.2435, 0.2616))
    ])
    
    # Clean test set
    test_set = CIFAR10(root='./data', train=False, 
                      download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    
    all_results = {}
    for eta in noise_rates:
        # Load noisy dataset
        train_set = CIFAR10(root='./data', train=True, 
                           download=True, transform=transform)
        train_set = add_symmetric_noise(train_set, eta)
        train_loader = DataLoader(train_set, batch_size=128, 
                                shuffle=True, num_workers=4)
        
        # Train and evaluate
        trainer = NoiseRobustTrainer(train_loader, test_loader)
        results = trainer.run_experiment(eta)
        all_results[f'η={eta}'] = results
        
    return all_results

# Visualization
def plot_results(results):
    plt.figure(figsize=(12, 5))
    
    # Accuracy vs Noise Rate
    plt.subplot(1, 2, 1)
    for loss_name in ['CE', 'NCE', 'FL', 'NFL']:
        accs = [results[f'η={eta}'][loss_name]['test_acc'][-1] 
               for eta in [0.2, 0.4, 0.6, 0.8]]
        plt.plot([0.2, 0.4, 0.6, 0.8], accs, marker='o', label=loss_name)
    plt.xlabel('Noise Rate (η)')
    plt.ylabel('Test Accuracy')
    plt.title('Final Accuracy vs Noise Rate')
    plt.legend()
    
    # Training Curves for η=0.6
    plt.subplot(1, 2, 2)
    eta = 0.6
    for loss_name in ['CE', 'NCE']:
        plt.plot(results[f'η={eta}'][loss_name]['test_acc'], 
                label=f'{loss_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title(f'Training Progress (η={eta})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('noise_robustness_results.png')
    plt.show()

# Main execution
if __name__ == "__main__":
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiments
    results = run_experiments()
    
    # Save results
    torch.save(results, 'noise_robustness_results.pt')
    
    # Generate plots
    plot_results(results)
