import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from cifar10_noisy import add_symmetric_noise  # Your existing noise function
from torchvision import transforms
from torchvision.datasets import CIFAR10
# Define Normalized Loss Functions
class NormalizedCrossEntropy(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        ce = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        
        # Compute sum of CE over all classes
        batch_size, num_classes = logits.shape
        all_targets = torch.arange(num_classes, device=logits.device).repeat(batch_size, 1)
        sum_ce = nn.CrossEntropyLoss(reduction='none')(logits.repeat_interleave(num_classes, 0), 
                                                     all_targets.view(-1))
        sum_ce = sum_ce.view(batch_size, num_classes).sum(dim=1)
        
        return (ce / (sum_ce + self.eps)).mean()

class NormalizedFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, eps=1e-8):
        super().__init__()
        self.gamma = gamma
        self.eps = eps
        
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        focal = (1 - probs)**self.gamma * (-torch.log(probs + self.eps))
        
        # Compute focal loss for true classes
        fl_true = focal[torch.arange(focal.size(0)), targets]
        
        # Compute sum over all classes
        sum_fl = focal.sum(dim=1)
        
        return (fl_true / (sum_fl + self.eps)).mean()

# Define Model Architecture (ResNet-18)
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        return self.classifier(x)

# Training Function
def train_model(loss_fn, loss_name, noise_rate=0.6, epochs=120):
    # Data preparation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_set = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Add noise with fixed seed for reproducibility
    noisy_train, _, _, eta = add_symmetric_noise(train_set, seed=42)
    
    # Create data loaders
    train_loader = DataLoader(noisy_train, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    
    # Initialize model and optimizer
    model = ResNet18().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate on test set
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
                
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%")
    
    print(f"\n{loss_name} Final Results:")
    print(f"η: {eta:.2f} | Best Test Accuracy: {best_acc:.2f}%")
    return best_acc

# Compare all loss functions
loss_functions = {
    "CE": nn.CrossEntropyLoss(),
    "FL": NormalizedFocalLoss(gamma=2.0),  # Vanilla FL (unnormalized)
    "NCE": NormalizedCrossEntropy(),
    "NFL": NormalizedFocalLoss(gamma=2.0)
}

results = {}
for loss_name, loss_fn in loss_functions.items():
    print(f"\n=== Training with {loss_name} ===")
    acc = train_model(loss_fn, loss_name)
    results[loss_name] = acc

# Generate comparison report
print("\n=== Final Comparison ===")
print(f"{'Loss Function':<10} | {'Test Accuracy (%)':>15}")
print("-" * 30)
for name, acc in results.items():
    print(f"{name:<10} | {acc:15.2f}")

# Visualization of training curves (add code to track and plot metrics)
