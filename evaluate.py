import pandas
import torch
import numpy

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import ResNet34
from dataset import dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet34()

#加载模型
parameters = torch.load('./model/ResNet34_10_Epoch.pth', map_location=torch.device(device))
model.load_state_dict(parameters)
model.to(device)

data_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
])
test_data = dataset("./dataset/test", data_transform, train=False)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
torch.set_grad_enabled(False)
model.eval()
result_list = numpy.zeros([test_data.__len__(), 2], dtype=int)
step = 0

for img, name in tqdm(test_loader):
    pred = model(img.to(device, torch.float))
    label = pred.argmax(dim=1)
    name = int(name[0])
    label = int(label)
    result_list[step, 0] = name
    result_list[step, 1] = label
    step += 1

result_list = result_list[result_list[:, 0].argsort()]
header = ["id", "label"]
csv_data = pandas.DataFrame(columns=header, data=result_list)
csv_data.to_csv(("./result/result.csv"), encoding="utf-8", index=False)
