import tensorflow as tf
from tensorflow.keras import layers
import sens2vec
from gensim.models import KeyedVectors
import numpy as np


class textCNN(object):

    def __init__(self,vocab_size,seq_length,embedding_size,kernel_size,embedding_matrix):
        '''
        初始化
        :param vocab_size: vocab的大小
        :param seq_length: 句子长度
        :param embedding_size: embedding的大小
        :param kernel_size: 卷积核的list
        :param embedding_matrix: 矩阵
        '''
        self.vocab_size=vocab_size
        self.embedding_size=embedding_size
        self.seq_length=seq_length
        self.kernel_size=kernel_size
        self.embedding_matrix=embedding_matrix

    def get_model(self):
        '''
        构建模型
        :return: 返回模型
        '''
        input=tf.keras.Input(shape=(self.seq_length,self.embedding_size,))
        '''embedding=layers.Embedding(input_dim=self.vocab_size,output_dim=self.embedding_size,
                                   input_length=self.seq_length)(input)'''
        convs=[]
        for kernel in self.kernel_size:
            c=layers.Conv1D(filters=2,kernel_size=kernel,activation=tf.nn.relu)(input)
            c=layers.GlobalMaxPooling1D()(c)
            convs.append(c)

        x=layers.Concatenate()(convs)

        output=layers.Dense(5,activation=tf.nn.softmax)(x)
        model=tf.keras.Model(inputs=input,outputs=output)
        return model

def get_embedding_matrix():
    '''
    如果要是word2vec得到的词向量矩阵，这个是必须的
    :return: vocab的embedding matrix
    '''
    w2v_model = KeyedVectors.load_word2vec_format(sens2vec.vocab_txt_file)
    vocab_list=[word for word,Vocab in w2v_model.wv.vocab.items()]
    embeddings_matrix = np.zeros((len(vocab_list), w2v_model.vector_size))

    for i in range(len(vocab_list)):
        word = vocab_list[i]  # 每个词语
        embeddings_matrix[i]=w2v_model.wv[word]  # 词向量矩阵
    return  embeddings_matrix


if __name__=="__main__":
    w2v_model = KeyedVectors.load_word2vec_format(sens2vec.vocab_txt_file)
    embedding_matrix=get_embedding_matrix()
    a=textCNN(len(list(w2v_model.wv.vocab.items())),64,128,[2,3,5],embedding_matrix)
    model=a.get_model()
    model.summary()


    model.compile(optimizer="Adam",
                  loss="categorical_crossentropy",
                  metric=["accuracy"])

    x_train,y_train=sens2vec.json2vec()
    x_train=np.array(x_train)
    y_train=tf.expand_dims(y_train,1)


    print(x_train.shape)
    print(y_train.shape)

    history=model.fit(x_train,y_train,batch_size=256,epochs=100,validation_split=0.3)
    model.save("textCNN.model",save_format="tf")
    #print(history.history["val_acc"])




