from flask import Flask, redirect, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from model import ResNet34
from torchvision import transforms

import torch
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './upload'

def predict(img_path: str) -> str:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet34()
    # 加载模型
    parameters = torch.load('./ResNet34_10_Epoch.pth', map_location=torch.device(device))
    model.load_state_dict(parameters)
    model.to(device)
    # 数据预处理
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    img = transform(img)
    pred = model(img.to(device, torch.float).unsqueeze(0))
    label = int(pred.argmax(dim=1))
    if label == 1:
        return "Dog"
    else:
        return "Cat"

def removeImage():
    if len(os.listdir("./upload")) > 100:
        os.system("rm -f ./upload/*")

@app.route("/", methods=['GET'])
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def upload():
    removeImage()
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    # 小于10MB
    if file.content_length > 10240:
        return "Image too large!"
    filename = secure_filename(file.filename)
    savepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(savepath)
    result = predict(savepath)
    return render_template("index.html", result=result, link=savepath)

@app.route("/upload/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run()

