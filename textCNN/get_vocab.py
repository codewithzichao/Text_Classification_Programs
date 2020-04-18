import json
import jieba
import tqdm
import gensim
from gensim.models import Word2Vec
from gensim.models import word2vec
import multiprocessing

train_data_file = "baike_qa2019/baike_qa_train.json"
my_train_data_file = "baike_qa2019/my_train.json"
stop_word_file = "stopword.txt"

def get_splited_corpus(file):
    '''
    得到使用jieba切分后的corpus与label
    :param file: 是处理的文件的路径
    :return: splited corpus
    '''
    stop_word_list = open(stop_word_file, "r", encoding="utf-8").read().split('\n')
    train_data = open(file, "r", encoding="utf-8").readlines()
    corpus_splited = []
    label = []

    for line in train_data:
        line_data_temp = json.loads(line)
        title_temp = line_data_temp["title"]
        title_temp_seg = list(jieba.cut(title_temp))
        line_temp = []
        for i in title_temp_seg:
            if i in stop_word_list:
                continue
            else:
                line_temp.append(i)

        if len(line_temp) != 0:
            corpus_splited.append(line_temp)

            category_temp = line_data_temp["category"][:2]
            label.append(category_temp)

    return corpus_splited, label


def get_vocab():
    '''
    得到vocab
    :return: 不返回，主要是得到vocab
    '''

    (corpus_splited,label)=get_splited_corpus(my_train_data_file)

    w2v_model=Word2Vec(corpus_splited,size=128,window=5,min_count=0,workers=multiprocessing.cpu_count())

    w2v_model.wv.save_word2vec_format("vocab.txt")


if __name__=="__main__":
    get_vocab()





