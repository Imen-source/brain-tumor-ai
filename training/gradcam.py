import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224  # keep consistent with your model


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        """
        input_tensor: torch.Tensor of shape (1, 3, 224, 224)
        class_idx: int, target class index
        Returns: overlay image as PIL.Image
        """
        input_tensor.requires_grad = True
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        # Convert gradients and activations to numpy
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]

        # Compute weights
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # avoid divide by zero

        # Convert input tensor to image
        # Convert input tensor to image
        img_np = input_tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)
        img_np = np.uint8(img_np * 255)


        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay heatmap on image
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        overlay_pil = Image.fromarray(overlay)

        return overlay_pil


def generate_gradcam(model, input_tensor, class_idx):
    """
    Simple function to use in FastAPI.
    """
    gradcam = GradCAM(model, model.layer4[-1])  # last layer of ResNet
    overlay_image = gradcam.generate(input_tensor, class_idx)
    return overlay_image


def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_bytes).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor
