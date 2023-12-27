import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms

model = torchvision.models.densenet201(weights=torchvision.models.densenet.DenseNet201_Weights.IMAGENET1K_V1)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
model.eval()

im = Image.open('bucket.jpeg')

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(im)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

# print(f'IM: {im}')
# print(f'input_tensor: {input_tensor}')
# print(f'input_tensor.shape: {input_tensor.shape}')
# print(f'input_batch: {input_batch}')
# print(f'input_batch.shape: {input_batch.shape}')

with torch.no_grad():
    output = model(input_batch)

# print(f'output: {output}')
# print(f'output.shape: {output.shape}')

probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())