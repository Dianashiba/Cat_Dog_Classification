# 随机抽取图片作为验证集
import os, shutil, random, re

def moveFile(pickNum):
    for x in ['cat', 'dog']:
        train_path = "./train/"
        val_path = "./validation/"
        img = os.listdir(train_path)
        img_list = [f for f in img if re.match(x, f)]
        sample = random.sample(img_list, pickNum)
        for name in sample:
            shutil.move(train_path+name, val_path+name)
    return

if __name__ == "__main__":
    if os.listdir('./validation/') == []:
        moveFile(1250)




