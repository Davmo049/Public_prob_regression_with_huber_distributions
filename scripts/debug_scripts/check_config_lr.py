from ConfigParser.ConfigParser import TrainingSettings, TrainingType
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    config_file = 'configurations/resnet_101_wflw.json'
    with open(config_file, 'rb') as f:
        config_dict = json.load(f)
    train_settings = TrainingSettings.deserialize(config_dict)
    lr_schedule = train_settings.lr_schedule
    num_epochs = train_settings.lr_schedule.num_epochs()
    x = []
    y = []
    for i in range(num_epochs):
        x.append(i)
        cur_lr = lr_schedule.get_learning_rate(i)
        y.append(cur_lr)
    plt.plot(x,y)
    plt.show()

if __name__ == '__main__':
    main()
