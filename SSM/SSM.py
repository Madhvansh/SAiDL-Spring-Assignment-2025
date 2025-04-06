import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import math
import torch.nn.functional as F
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# 1. S4D Kernel Implementation
class S4DKernel(nn.Module):
    def __init__(self, d_model, n, l_max, dt=0.1, discretization='bilinear'):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.dt = dt
        self.discretization = discretization

        self.L = nn.Parameter(torch.ones(d_model, n) * 0.5)
        self.P_left = nn.Parameter(torch.empty(d_model, n, 2).normal_(0, 0.01))
        self.P_right = nn.Parameter(torch.empty(d_model, 2, n).normal_(0, 0.01))
        self.B = nn.Parameter(torch.randn(d_model, n) / math.sqrt(n))
        self.C = nn.Parameter(torch.randn(d_model, n) / math.sqrt(n))
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.log(dt * torch.ones(d_model, 1)))

    def _compute_kernel(self, L):
        L_mat = torch.diag_embed(torch.exp(self.L))  # [d_model, n, n]
        P = torch.bmm(self.P_left, self.P_right)     # [d_model, n, n]
        A = -L_mat + P

        dt = torch.exp(self.log_dt)  # [d_model, 1]
        A = A * dt.unsqueeze(-1)     # [d_model, n, n]
        B = self.B * dt              # [d_model, n]

        I = torch.eye(self.n, device=A.device).unsqueeze(0)  # [1, n, n]
        if self.discretization == 'bilinear':
            dA = torch.linalg.solve(I - A/2, I + A/2)  # [d_model, n, n]
            dB = torch.linalg.solve(I - A/2, B.unsqueeze(-1))  # [d_model, n, 1]
        elif self.discretization == 'zoh':
            dA = torch.matrix_exp(A)  # [d_model, n, n]
            dB = torch.linalg.solve(A, (dA - I)) @ B.unsqueeze(-1)  # [d_model, n, 1]

        kernel = torch.zeros(self.d_model, self.l_max, device=A.device)
        state = dB  # [d_model, n, 1]
        
        for t in range(self.l_max):
            # Compute C @ state for each d_model channel
            kernel[:, t] = torch.bmm(self.C.unsqueeze(1), state).squeeze(-1).squeeze(-1)  # [d_model] <- [d_model, 1, n] @ [d_model, n, 1]
            state = torch.bmm(dA, state)  # [d_model, n, 1]
        
        return kernel

    def forward(self, x):
        batch, L, _ = x.shape
        x = x.transpose(1, 2)  # [batch, d_model, L]
        
        kernel = self._compute_kernel(L)
        
        kernel_f = torch.fft.rfft(kernel, n=2*self.l_max)
        x_f = torch.fft.rfft(x, n=2*self.l_max)
        y = torch.fft.irfft(x_f * kernel_f, n=2*self.l_max)[..., :L]
        
        y = y + self.D.view(1, -1, 1) * x  # [batch, d_model, L]
        return y.transpose(1, 2)  # [batch, L, d_model]

# 2. Dataset
class PatchedCIFAR10(Dataset):
    def __init__(self, root='./data', train=True):
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.RandomErasing(p=0.5, value='random')  # Adjust for consistency
        ]) if train else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        
        self.dataset = torchvision.datasets.CIFAR10(
            root=root, train=train, download=True, transform=transform)

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

# 3. S4 Block
class S4Block2D(nn.Module):
    def __init__(self, d_model, n, l_max, dropout=0.2, glu=True):
        super().__init__()
        self.s4 = S4DKernel(d_model, n, l_max)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = glu
        
        if glu:
            self.proj = nn.Linear(d_model, 2*d_model)
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        residual = x  # [batch, L, d_model]
        x = self.norm(x)
        x = self.s4(x)  # [batch, L, d_model]
        x = self.proj(x)  # [batch, L, 2*d_model] or [batch, L, d_model]
        if self.glu:
            x = F.glu(x, dim=-1)  # [batch, L, d_model]
        return residual + self.dropout(x)

# 4. Enhanced S4 Model with 2D Awareness
class S4Model2D(nn.Module):
    def __init__(self, d_model=512, n_layers=12, n=64, patch_size=8, 
                 dropout=0.2, l_max=16, glu=True):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (32 // patch_size)**2

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, d_model//2, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model//2, d_model, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
        )
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        self.blocks = nn.ModuleList([
            S4Block2D(d_model, n, l_max, dropout=dropout, glu=glu)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)
        
        for block in self.blocks:
            nn.utils.spectral_norm(block.proj)

    def forward(self, x):
        # x: [B, 3, 32, 32]
        x = self.patch_embed(x)  # [B, d_model, num_patches_h, num_patches_w]
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # [B, num_patches, d_model]
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x.mean(dim=1))
        return self.head(x)

# 5. Training Configuration with Warmup
def train_advanced():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(PatchedCIFAR10(), batch_size=128, 
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(PatchedCIFAR10(train=False), 
                             batch_size=256, shuffle=False, num_workers=4)

    model = S4Model2D(d_model=512, n_layers=12).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, 
                            weight_decay=0.1, betas=(0.9, 0.98))
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-4,
        total_steps=200 * len(train_loader),
        pct_start=0.05,
        anneal_strategy='cos'
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    for epoch in range(200):
        model.train()
        total_loss = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == targets).sum().item()
        
        acc = 100 * correct / len(test_loader.dataset)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 's4_2d_best.pth')
        
        print(f'Epoch {epoch+1}/200 | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.2f}%')

    print(f'Best Test Accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    train_advanced()
