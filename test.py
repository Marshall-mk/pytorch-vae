import torch
from torchvision.utils import save_image
from models.cvae import ConditionalVAE
import yaml
import os
import torchvision.transforms as transforms
# Load the configuration file
config_path = 'configs/cvae.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load the trained model
model = ConditionalVAE(**config['model_params'])
checkpoint = torch.load('logs/ConditionalVAE/version_0/checkpoints/last.ckpt')
new_state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(new_state_dict)
model.eval()

# Generate synthetic images for each class
num_samples = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# for class_idx in range(3):
#     labels = torch.zeros(num_samples, 3).to(device)
#     labels[:, class_idx] = 1
#     with torch.no_grad():
#         samples = model.sample(num_samples, device, labels=labels)
#         save_image(samples, f'synthetic_class_{class_idx}.png', nrow=5, normalize=True)

# Define the transformation to upscale images
upscale_transform = transforms.Resize((224, 224))

for class_idx in range(3):
    # Create a directory for the class if it doesn't exist
    class_dir = f'class_{class_idx}'
    os.makedirs(class_dir, exist_ok=True)

    labels = torch.zeros(num_samples, 3).to(device)
    labels[:, class_idx] = 1
    with torch.no_grad():
        samples = model.sample(num_samples, device, labels=labels)
        
        # Save each individual image as a PNG file
        for i in range(num_samples):  # {{ edit_4 }}
            upscaled_image = upscale_transform(samples[i])  # Upscale the image
            save_image(upscaled_image, os.path.join(class_dir, f'synthetic_class_{class_idx}_image_{i}.png'), normalize=True)