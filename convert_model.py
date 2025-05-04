import torch
from torch import nn, optim
from torchvision import datasets,transforms
import hls4ml

model = nn.Sequential(nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            # nn.Softmax(dim=1)
           )
model.load_state_dict(torch.load('model.pth'))
model.eval() 
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                               ])

trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
dataiter = iter(trainloader)
images, labels = next(dataiter)
img = images[0]
config = hls4ml.utils.config_from_pytorch_model(
     model,
     granularity='model',
     default_precision='fixed<16,6>',
     backend='VivadoAccelerator')

hls_model = hls4ml.converters.convert_from_pytorch_model(
     model,
     hls_config=config,
     output_dir="output",
     backend='VivadoAccelerator',
     io_type='io_stream',
     board='pynq-z2',
     project_name='dca_3',
     input_shape=(1, 784),
  )

# hls_model.compile()
hls_model.build(bitfile=True)
numpy_img = img.numpy()
hls_prediction = hls_model.predict(numpy_img)
print(type(hls_prediction))


