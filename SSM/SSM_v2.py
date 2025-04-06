import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import math
import torch.nn.functional as F

class SequentialCIFAR10(Dataset):
    def __init__(self, root='./data', train=True, transform=None):
        self.cifar = torchvision.datasets.CIFAR10(
            root=root, 
            train=train, 
            download=True,
            transform=transform
        )
        
    def __len__(self):
        return len(self.cifar)
        
    def __getitem__(self, index):
        img, label = self.cifar[index]
        img = img.permute(1, 2, 0).flatten(0, 1)  # [1024, 3]
        return img, label

class S4DKernel(nn.Module):
    def __init__(self, d_model, n, l_max, dt=1.0):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.dt = dt

        # Complex parameter initialization
        self.A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, n//2)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n//2).float().repeat(d_model, 1))
        
        # Complex B and C parameters
        B_real = torch.randn(d_model, n//2) * 0.5
        B_imag = torch.randn(d_model, n//2) * 0.5
        self.B = nn.Parameter(torch.complex(B_real, B_imag))
        
        C_real = torch.randn(d_model, n//2) / math.sqrt(n)
        C_imag = torch.randn(d_model, n//2) / math.sqrt(n)
        self.C = nn.Parameter(torch.complex(C_real, C_imag))
        
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u):
        batch, L, _ = u.shape

        # Construct diagonal matrix
        A = -torch.exp(self.A_real) + 1j * self.A_imag
        
        # Discretization
        I = torch.eye(self.n//2, device=u.device)
        dA = (I + A.unsqueeze(-1) * self.dt/2) / (I - A.unsqueeze(-1) * self.dt/2)

        # Kernel computation with complex numbers
        power = torch.linalg.matrix_power(dA, L)
        kernel = torch.einsum('dnh,dh->dn', 
                            self.C.unsqueeze(-1) * (I - power), 
                            self.B).real  # Convert to real after computation

        # FFT convolution
        u = u.transpose(1, 2).float()
        fft_size = 2 * L
        
        kernel_f = torch.fft.rfft(kernel, n=fft_size)
        u_f = torch.fft.rfft(u, n=fft_size)
        
        y = torch.fft.irfft(u_f * kernel_f.unsqueeze(0), n=fft_size)[..., :L]
        y = y + self.D.view(1, -1, 1) * u
        
        return y.transpose(1, 2)

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
        return self.activation(self.norm(residual + x))

class S4Model(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n=64, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)
        self.layers = nn.ModuleList([
            S4Layer(d_model, n, 1024, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, 10)
        
        # Weight initialization
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(self.norm(x.mean(dim=1)))

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data pipeline
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = SequentialCIFAR10(transform=train_transform)
    test_dataset = SequentialCIFAR10(train=False, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model and training setup
    model = S4Model().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(50):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == targets).sum().item()
        
        acc = 100 * correct / len(test_dataset)
        scheduler.step()
        print(f'Epoch {epoch+1}/50 | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%')

if __name__ == "__main__":
    train_model()