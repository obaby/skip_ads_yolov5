# 模型效果测试
import os

with open('data/ImageSets/test.txt', "r") as f:
    for line in f.readlines():
        cmd = 'python detect_ads.py --source data/images/' +line.strip()+ '.jpg'
        os.system(cmd)