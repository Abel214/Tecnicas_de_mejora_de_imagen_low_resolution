import torch
import os
from utils import training_device, read_model, read_cv2, image_to_tensor_cv2, tensor_to_image_cv2, insert_filename_ending
from model import SRNet

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../model_gan')
test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../test_data')

# Read model
device = training_device(True)
model = SRNet().to(device)
_, _ = read_model(os.path.join(model_path, 'model.pt'), model)
model.half().eval()

# Define inference function
def inference(filename):
    image = read_cv2(filename)

    image_tensor = image_to_tensor_cv2(image).half().to(device)
    with torch.no_grad():
        output = model(image_tensor)
    output = tensor_to_image_cv2(output, insert_filename_ending(filename, '_sr'))

if __name__ == '__main__':
    test_filename = os.path.join(test_data_path, 'test.png')
    inference(test_filename)
