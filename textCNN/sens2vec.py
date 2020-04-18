from gensim.models import KeyedVectors
import get_vocab
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

train_data_file = "baike_qa2019/baike_qa_train.json"
my_train_data_file = "baike_qa2019/my_train.json"
stop_word_file = "stopword.txt"
vocab_txt_file="vocab.txt"
seq_length=64
embedding_size=128

def json2vec():

    w2v_model=KeyedVectors.load_word2vec_format(vocab_txt_file)
    corpus_splited,label=get_vocab.get_splited_corpus(my_train_data_file)

    text_input=[]
    for sentence in corpus_splited:
        text_input.append(np.array([w2v_model.wv[word] for word in sentence if word in w2v_model.wv.vocab]))
    print(text_input.__len__(),text_input[0].shape)


    text_input_vec=[]
    for item in tqdm(text_input):
        if(int(item.shape[0])<seq_length):
            while(int(item.shape[0])<seq_length):

                temp=np.array([[0]*embedding_size])

                item=np.concatenate((item,temp))

        elif(item.shape[0]>seq_length):
            item=item[:seq_length]

        text_input_vec.append(item)

    text_input_vec=np.array(text_input_vec)

    encoder=LabelEncoder()
    label=np.array(label)
    label=encoder.fit_transform(label)
    print(label[:10])
    label_vec=to_categorical(label)

    print(text_input_vec.shape,label_vec.shape)


    return text_input_vec,label_vec

if __name__=="__main__":
    json2vec()






