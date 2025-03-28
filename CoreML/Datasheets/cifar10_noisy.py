import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms

def add_symmetric_noise(dataset, seed=42):
    """Add symmetric label noise to CIFAR-10 dataset with random η ∈ [0.2, 0.8]"""
    rng_eta = np.random.default_rng()
    eta = rng_eta.uniform(0.2, 0.8)
    
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    noise_matrix = np.zeros((num_classes, num_classes))
    original_targets = targets.copy()

    for orig_class in range(num_classes):
        class_indices = np.where(original_targets == orig_class)[0]
        n_samples = len(class_indices)
        n_noisy = int(eta * n_samples)
        
        if n_noisy > 0:
            noisy_idx = np.random.choice(class_indices, n_noisy, replace=False)
            new_labels = np.random.choice(
                [c for c in range(num_classes) if c != orig_class],
                size=n_noisy
            )
            targets[noisy_idx] = new_labels
            
            # Track both correct and corrupted samples
            noise_matrix[orig_class, orig_class] = n_samples - n_noisy
            for new_cls in new_labels:
                noise_matrix[orig_class, new_cls] += 1

    dataset.targets = targets.tolist()
    return dataset, noise_matrix, original_targets, eta

def visualize_noisy_samples(dataset, original_targets, noisy_targets, class_names, num_samples=1):
    """Visualize original vs noisy labels for each class"""
    plt.figure(figsize=(20, 10))
    
    for cls_idx in range(len(class_names)):
        # Get corrupted samples for current class
        cls_corrupted = np.where((original_targets == cls_idx) & (original_targets != noisy_targets))[0]
        
        if len(cls_corrupted) > 0:
            idx = np.random.choice(cls_corrupted)
            
            ax = plt.subplot(2, 5, cls_idx + 1)
            
            # Display image
            img = dataset.data[idx]
            plt.imshow(img)
            
            # Annotate with labels
            true_label = class_names[original_targets[idx]]
            noisy_label = class_names[noisy_targets[idx]]
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
noisy_targets = np.array(noisy_dataset.targets)

# Print statistics
total_samples = len(original_targets)
corrupted_count = (original_targets != noisy_targets).sum()
print(f"Generated η: {eta:.2f}")
print(f"Total samples: {total_samples}")
print(f"Corrupted samples: {corrupted_count} ({corrupted_count/total_samples:.2%})")

print("\nClass-wise Noise Statistics:")
for cls_idx, cls_name in enumerate(train_dataset.classes):
    total_cls = np.sum(original_targets == cls_idx)
    correct_cls = noise_matrix[cls_idx, cls_idx]
    corrupted_cls = total_cls - correct_cls
    print(f"{cls_name:>10}: {corrupted_cls:5} corrupted ({corrupted_cls/total_cls:.2%})")

# Visualize samples with label noise
visualize_noisy_samples(
    train_dataset,
    original_targets,
    noisy_targets,
    class_names=train_dataset.classes
)
