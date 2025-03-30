import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from cifar10_noisy import add_symmetric_noise  # Your existing noise function
from torchvision import transforms
from torchvision.datasets import CIFAR10

# 1. Define Normalized Loss Functions ==========================================
class NormalizedCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        ce = -torch.log(probs[torch.arange(inputs.size(0)), targets] + 1e-10)
        norm = torch.sum(-torch.log(probs + 1e-10), dim=1)
        return torch.mean(ce / norm)

class NormalizedFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        probs = torch.softmax(inputs, dim=1)
        focal = (1 - probs[torch.arange(inputs.size(0)), targets])**self.gamma * \
               -torch.log(probs[torch.arange(inputs.size(0)), targets] + 1e-10)
        norm = torch.sum((1 - probs)**self.gamma * -torch.log(probs + 1e-10), dim=1)
        return torch.mean(focal / norm)

# 2. Define Model Architecture ================================================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.linear = nn.Linear(256, num_classes)
        
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.AdaptiveAvgPool2d((1,1))(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)

# 3. Training Pipeline ========================================================
def train_model(eta, loss_fn, num_epochs=100):
    # Load dataset with specific noise level
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load clean dataset and add noise
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    noisy_set, _, _, _ = add_symmetric_noise(train_set, eta=eta)
    
    train_loader = DataLoader(noisy_set, batch_size=128, shuffle=True, num_workers=2)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    # Initialize model and optimizer
    model = ResNet().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    history = {'train_loss': [], 'test_acc': []}
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        avg_loss = total_loss / len(train_loader.dataset)
        test_acc = 100 * correct / total
        history['train_loss'].append(avg_loss)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Acc: {test_acc:.2f}%')
    
    return history

# 4. Run Experiments ==========================================================
def run_experiments():
    eta_values = [0.2, 0.5, 0.8]
    loss_functions = {
        'CE': nn.CrossEntropyLoss(),
        'FL': NormalizedFocalLoss(),  # Use same class with gamma=0
        'NCE': NormalizedCrossEntropy(),
        'NFL': NormalizedFocalLoss(gamma=2)
    }
    
    results = {}
    for eta in eta_values:
        print(f"\n=== Training with η={eta} ===")
        eta_results = {}
        
        for loss_name, loss_fn in loss_functions.items():
            print(f"\nTraining with {loss_name}...")
            history = train_model(eta, loss_fn)
            eta_results[loss_name] = {
                'best_acc': max(history['test_acc']),
                'history': history
            }
            
        results[eta] = eta_results
    
    # Plot results
    for eta in eta_values:
        plt.figure(figsize=(12, 4))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        for loss_name in loss_functions:
            plt.plot(results[eta][loss_name]['history']['train_loss'], 
                    label=f'{loss_name}')
        plt.title(f'Training Loss (η={eta})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy curves
        plt.subplot(1, 2, 2)
        for loss_name in loss_functions:
            plt.plot(results[eta][loss_name]['history']['test_acc'], 
                    label=f'{loss_name}')
        plt.title(f'Test Accuracy (η={eta})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    # Print final comparison
    print("\nFinal Comparison:")
    print(f"{'Loss':<6} {'η=0.2':<8} {'η=0.5':<8} {'η=0.8':<8}")
    for loss_name in loss_functions:
        accs = [f"{results[eta][loss_name]['best_acc']:.2f}%" for eta in eta_values]
        print(f"{loss_name:<6} {' '.join(accs)}")

if __name__ == "__main__":
    torch.manual_seed(42)
    run_experiments()