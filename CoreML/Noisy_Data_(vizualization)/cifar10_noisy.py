import torch
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms

def add_symmetric_noise(dataset, seed=42):
    """Add symmetric label noise to CIFAR-10 dataset with random η ∈ [0.2, 0.8]"""
    # Generate random η using PyTorch
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

def visualize_noisy_samples(dataset, original_targets, noisy_targets, class_names, num_samples=1):
    """Visualize original vs noisy labels for each class using PyTorch tensors"""
    plt.figure(figsize=(20, 10))
    
    # Convert to tensors if necessary
    if not isinstance(original_targets, torch.Tensor):
        original_targets = torch.tensor(original_targets)
    if not isinstance(noisy_targets, torch.Tensor):
        noisy_targets = torch.tensor(noisy_targets)
    
    for cls_idx in range(len(class_names)):
        # Get corrupted samples for current class
        cls_mask = (original_targets == cls_idx) & (original_targets != noisy_targets)
        cls_corrupted = cls_mask.nonzero(as_tuple=True)[0]
        
        if cls_corrupted.size(0) > 0:
            # Randomly select a sample using PyTorch
            rand_idx = torch.randint(0, cls_corrupted.size(0), (1,))
            idx = cls_corrupted[rand_idx].item()
            
            ax = plt.subplot(2, 5, cls_idx + 1)
            
            # Display image (CIFAR-10 stores data as numpy array)
            img = dataset.data[idx]
            plt.imshow(img)
            
            # Annotate with labels
            true_label = class_names[original_targets[idx].item()]
            noisy_label = class_names[noisy_targets[idx].item()]
            plt.title(f"True: {true_label}\nNoisy: {noisy_label}", color='red', fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Load CIFAR-10 dataset
train_dataset = CIFAR10(
    root='./data', 
    train=True, 
    download=True,
    transform=transforms.ToTensor()
)

# Add noise with random η
noisy_dataset, noise_matrix, original_targets, eta = add_symmetric_noise(train_dataset)
noisy_targets = torch.tensor(noisy_dataset.targets)

# Print statistics
total_samples = original_targets.size(0)
corrupted_count = (original_targets != noisy_targets).sum().item()
print(f"Generated η: {eta:.2f}")
print(f"Total samples: {total_samples}")
print(f"Corrupted samples: {corrupted_count} ({corrupted_count/total_samples:.2%})")

print("\nClass-wise Noise Statistics:")
for cls_idx, cls_name in enumerate(train_dataset.classes):
    total_cls = (original_targets == cls_idx).sum().item()
    correct_cls = noise_matrix[cls_idx, cls_idx].item()
    corrupted_cls = total_cls - correct_cls
    print(f"{cls_name:>10}: {corrupted_cls:5} corrupted ({corrupted_cls/total_cls:.2%})")

# Visualize samples with label noise
visualize_noisy_samples(
    train_dataset,
    original_targets,
    noisy_targets,
    class_names=train_dataset.classes
)
