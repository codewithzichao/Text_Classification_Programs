import jieba
import json

train_data_file = "baike_qa2019/baike_qa_train.json"
my_train_dat_file = "baike_qa2019/my_train.json"
stop_word_file = "stopword.txt"

# 总共有492个类别，在这里，为了方便期间，只选用了5个类别
target_class = {"教育": 0, "健康": 0, "生活": 0, "娱乐": 0, "游戏": 0}
per_class_num = 6000
total_num = per_class_num * 5


def get_train_data():
    '''
    得到想要的train_data，共5个类别，每一个类别有6000个样本，共30000个样本
    :return: 不返回值，主要是得到json文件
    '''
    total_num_temp = 0
    with open(train_data_file, "r", encoding="utf-8") as f1:
        with open(my_train_dat_file, "w", encoding="utf-8") as f2:
            for line in f1.readlines():
                data_temp = json.loads(line)
                class_temp = data_temp["category"][0:2]

                if class_temp in target_class and target_class[class_temp] < per_class_num:
                    # 注意，这里一定要使用ensure_ascii=False，要不然默认是ascii编码！
                    json_data = json.dumps(data_temp, ensure_ascii=False)
                    f2.write(json_data)
                    f2.write("\n")
                    target_class[class_temp] += 1
                    total_num_temp += 1

                    if (total_num_temp == total_num):
                        break


if __name__ == "__main__":
    get_train_data()
