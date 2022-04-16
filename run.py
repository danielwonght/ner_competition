import os

from src.lightning_module.pl_ner import *
from src.utils import file_reading, file_writing, convert_data_file
from src.lightning_module.config import Config
from sklearn.model_selection import train_test_split

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__": 
    random_split = False
    convert_data = False
    # predict_path = CUR_DIR + "/datasets/JDNER/train.txt"
    predict_path = CUR_DIR + "/datasets/JDNER/test.txt"
    
    if random_split:
        train_path = CUR_DIR + "/train_data/train.txt"
        train_data = file_reading(train_path)
        train_data, dev_data = train_test_split(train_data, test_size=0.1, shuffle=True)
    else:
        train_path = CUR_DIR + "/datasets/JDNER/train.txt"
        # train_path = CUR_DIR + "/outputs/test_result_train.txt"
        # train_path = CUR_DIR + "/train_data/train.txt"
        dev_path = CUR_DIR + "/datasets/JDNER/dev.txt"
        dev_data = file_reading(dev_path)
    
    if convert_data:
        new_train_path = CUR_DIR + "/datasets/JDNER/new_train.txt"
        convert_data_file(train_path, new_train_path)
        train_data = file_reading(new_train_path)
    else:
        train_data = file_reading(train_path)
        
    predict_data = file_reading(predict_path)
    print(predict_data[0])
    config = Config()
    label2id = FeatureConverter.generate_label2id()
    id2label = {v:k for k,v in label2id.items()}
    train(train_data, dev_data, config, label2id, id2label, gpus=[0], use_attack=True)
    predict_data = predict(predict_data, config)
    file_writing(predict_data, CUR_DIR + "/outputs/test_result_focal_roformer_pretrained_attack.txt")


    # import pandas as pd
    # unlabeled_path = "/opt/wekj/aimeng_huawei_cloud/pretrain/train_data/unlabeled_train_data.txt"
    # unlabeled_data = []
    # with open(unlabeled_path, 'r') as f:
    #     unlabeled_data = f.read().splitlines()
    #     unlabeled_data = [{"query":[xi if xi not in (""," ") else "[SPACE]" for xi in x]} for x in unlabeled_data]
    # print(f"len unlabeled data = {len(unlabeled_data)}")
    # # print(unlabeled_data)
    # unlabeled_data = predict(unlabeled_data, config)
    # unlabeled_data = pd.DataFrame(unlabeled_data)
    # unlabeled_data.to_csv("/opt/wekj/aimeng_huawei_cloud/pretrain/ner/datasets/JDNER/unlabeled_04.csv", index=None)
