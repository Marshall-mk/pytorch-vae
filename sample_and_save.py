import torch
from torchvision.utils import save_image
from models.cvae import ConditionalVAE
import yaml
import os
import torchvision.transforms as transforms

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

# Generate synthetic images for each class
num_samples = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the transformation to upscale images
upscale_transform = transforms.Resize((224, 224))

def sample_and_save(num_samples, output_folder='sampled_images'):
    os.makedirs(output_folder, exist_ok=True)
    for class_idx in range(3):
        class_dir = os.path.join(output_folder, f'{class_map[str(class_idx)]}')
        os.makedirs(class_dir, exist_ok=True)

        labels = torch.zeros(num_samples, 3).to(device)
        labels[:, class_idx] = 1
        with torch.no_grad():
            samples = model.sample(num_samples, device, labels=labels)
            
            # Save each individual image as a PNG file
            for i in range(num_samples):
                # upscaled_image = upscale_transform(samples[i])  # Upscale the image
                upscaled_image = samples[i]
                save_image(upscaled_image, os.path.join(class_dir, f'synthetic_class_{class_idx}_image_{i}.png'), normalize=True)

# Call the function to sample and save images
sample_and_save(num_samples)