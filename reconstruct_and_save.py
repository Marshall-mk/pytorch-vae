import torch
from torchvision.utils import save_image
from models.cvae import ConditionalVAE
import yaml
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class_map = {
    "0": "bcc",
    "1": "mel",
    "2": "scc"
}
# Load the configuration file
config_path = 'configs/cvae.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load the trained model
model = ConditionalVAE(**config['model_params'])
checkpoint = torch.load('logs/ConditionalVAE/version_1/checkpoints/last.ckpt')
new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(new_state_dict)
model.eval()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# define the initial transformation to downscale images to 64x64
data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
# Define the transformation to upscale images
upscale_transform = transforms.Resize((224, 224))

def reconstruct_and_save(data_loader, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for class_idx in range(3):
        class_dir = os.path.join(output_folder, f'{class_map[str(class_idx)]}')
        os.makedirs(class_dir, exist_ok=True)
        
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            # one hot encode the labels
            one_hot_labels = torch.zeros(len(labels), 3).to(device)
            one_hot_labels[:, class_idx] = 1  # Set the current class index to 1
            
            with torch.no_grad():
                recon_images = model.generate(images, labels=one_hot_labels)
                for i, recon_image in enumerate(recon_images):
                    if labels[i].item() == class_idx:  # Check if the label matches the current class
                        save_image(recon_image, os.path.join(class_dir, f'reconstructed_image_{batch_idx * len(images) + i}.png'), normalize=True)
                        
# Load train and validation datasets
train_dataset = ImageFolder(root='../../../dataset/3_class_train/', transform=data_transform)
val_dataset = ImageFolder(root='../../../dataset/3_class_val/', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Reconstruct and save train and validation data
reconstruct_and_save(train_loader, 'train_reconstructed')
reconstruct_and_save(val_loader, 'val_reconstructed')