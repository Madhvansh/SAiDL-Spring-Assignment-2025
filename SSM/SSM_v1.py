import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import math
import torch.nn.functional as F

class S4DKernel(nn.Module):
    def __init__(self, d_model, n, l_max, dt=0.1):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.dt = dt
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        # Initialize A using S4D-Lin method
        A_real = torch.log(torch.ones(self.d_model, self.n // 2) * 0.5)
        A_imag = torch.arange(self.n // 2).float() * math.pi
        A_imag = A_imag.expand(self.d_model, -1)
        
        self.A_real = nn.Parameter(A_real)
        self.A_imag = nn.Parameter(A_imag)
        
        # Initialize B, C as complex parameters
        B_real = torch.randn(self.d_model, self.n // 2)
        B_imag = torch.randn(self.d_model, self.n // 2)
        self.B_real = nn.Parameter(B_real)
        self.B_imag = nn.Parameter(B_imag)
        
        C_real = torch.randn(self.d_model, self.n // 2)
        C_imag = torch.randn(self.d_model, self.n // 2)
        self.C_real = nn.Parameter(C_real)
        self.C_imag = nn.Parameter(C_imag)
        
        # Initialize D
        self.D = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, u):
        batch, L, _ = u.shape
        
        # Construct complex parameters
        A_real = -torch.exp(self.A_real)  # Ensure stability
        A = torch.complex(A_real, self.A_imag)
        B = torch.complex(self.B_real, self.B_imag)
        C = torch.complex(self.C_real, self.C_imag)
        
        # Discretize
        A_discrete = torch.exp(A * self.dt)
        
        # Frequency domain computation
        freqs = torch.fft.rfftfreq(2 * L, 1 / L).to(A.device)
        omega = torch.exp(-2j * math.pi * freqs / L).view(1, 1, -1)
        
        # Reshape for broadcasting
        A_discrete = A_discrete.unsqueeze(-1)  # [d_model, n//2, 1]
        C = C.unsqueeze(-1)  # [d_model, n//2, 1]
        
        # Compute frequency response
        k_f = (C / (1 - omega * A_discrete)).sum(dim=1)  # [d_model, L+1]
        
        # Convert to time domain
        k = torch.fft.irfft(k_f, n=2*L)[:, :L]  # [d_model, L]
        
        # Apply convolution
        pad_left = (k.size(-1) - 1) // 2  # 511 for L=1024
        pad_right = k.size(-1) // 2       # 512 for L=1024
        
        u = u.transpose(1, 2)  # [batch, d_model, L]
        u_padded = F.pad(u, (pad_left, pad_right))
        
        y = F.conv1d(
            u_padded, 
            k.unsqueeze(1),  # [d_model, 1, L]
            bias=self.D,
            groups=self.d_model,
            padding=0
        )
        return y.transpose(1, 2) # [batch, L, d_model]

class S4Layer(nn.Module):
    def __init__(self, d_model, n, l_max, dropout=0.1):
        super().__init__()
        self.s4 = S4DKernel(d_model, n, l_max)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.s4(x)
        x = self.dropout(x)
        x = residual + x
        x = self.norm(x)
        return self.activation(x)

class S4Model(nn.Module):
    def __init__(self, d_input, d_model, d_output, n_layers, n, l_max, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        
        self.layers = nn.ModuleList([
            S4Layer(d_model, n, l_max, dropout) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_output)
    
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.output_proj(x)

class SequentialCIFAR10(Dataset):
    def __init__(self, root='./data', train=True):
        self.cifar = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        
    def __len__(self):
        return len(self.cifar)
        
    def __getitem__(self, index):
        img, label = self.cifar[index]
        img = img.permute(1, 2, 0).reshape(-1, 3)  # (32*32, 3)
        return img, label

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(test_loader), 100. * correct / total

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    d_model = 128
    n_layers = 4
    n = 64
    l_max = 1024
    lr = 1e-3
    weight_decay = 1e-5
    dropout = 0.1
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    train_dataset = SequentialCIFAR10(train=True)
    test_dataset = SequentialCIFAR10(train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model initialization
    model = S4Model(
        d_input=3,
        d_model=d_model,
        d_output=10,
        n_layers=n_layers,
        n=n,
        l_max=l_max,
        dropout=dropout
    ).to(device)

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%\n')

    # Save model
    torch.save(model.state_dict(), 's4_cifar10.pth')
