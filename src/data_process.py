# -*- coding:utf-8 -*-
import codecs
import json
import os
import random
import numpy as np
import pickle
import jieba

ConfDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "conf")
DataDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
UNKNOWN = "<OOV>"


def _save_vocab(dict, path):
    # save utf-8 code dictionary
    outfile = codecs.open(path, "w", encoding='utf-8')
    for k, v in dict.items():
        # k is unicode, v is int
        line = k + "\t" + str(v) + "\n"  # unicode
        outfile.write(line)
    outfile.close()


def load_vector_file(vector_file_path):
    vector_dicts = dict()
    with codecs.open(vector_file_path, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split(" ")
            vector_dicts[line[0]] = np.asarray(list(map(float, line[1:])), dtype=np.float32)
    return vector_dicts


def load_data(configs):
    corpus_dir = os.path.join(DataDirPath, configs["corpus_id"])
    text_dir = os.path.join(corpus_dir, "text")
    pinyin_dir = os.path.join(corpus_dir, "pinyin")
    text_file_list = os.listdir(text_dir)
    file_to_tag_list = []
    file_to_tag_dict = dict()
    for tag_idx, file_name in enumerate(text_file_list):
        file_to_tag_list.append((tag_idx, file_name.split(".")[0]))
        file_to_tag_dict[file_name] = tag_idx
    Data_lists = []
    for (tag, file_name) in file_to_tag_list:
        text_file = os.path.join(text_dir, file_name+".text")
        pinyin_file = os.path.join(pinyin_dir, file_name+".pinyin")
        if os.path.isfile(text_file) and os.path.isfile(pinyin_file):
            text_reader = codecs.open(text_file, mode='r', encoding='utf-8')
            pinyin_reader = codecs.open(pinyin_file, mode='r', encoding='utf-8')
            nums = 0
            for text_line, pinyin_line in zip(text_reader.readlines(), pinyin_reader.readlines()):
                if configs["cut_word"]:
                    item = (jieba.lcut(text_line.strip()), pinyin_line.strip().split(" "), tag)
                else:
                    item = ([ch for ch in text_line.strip()], pinyin_line.strip().split(" "), tag)
                Data_lists.append(item)
                nums += 1
            print(file_name+" tag:", tag, " data nums:", nums)
        else:
            raise ValueError(text_file+" or "+pinyin_file + " not exist")
    print("total data nums:", len(Data_lists))
    random.shuffle(Data_lists)
    return Data_lists, file_to_tag_dict


def data_vectorization(configs, data_lists):
    if configs["word_embedding_file"] != "None":
        pre_train_word_vector_dicts = load_vector_file(os.path.join(DataDirPath, configs["word_embedding_file"]))
    else:
        pre_train_word_vector_dicts = None
    if configs["pinyin_embedding_file"] != "None":
        pre_train_pinyin_vector_dicts = load_vector_file(os.path.join(DataDirPath, configs["pinyin_embedding_file"]))
    else:
        pre_train_pinyin_vector_dicts = None
    word_vector_dim = int(configs["word_embedding_dim"])
    pinyin_vector_dim = int(configs["pinyin_embedding_dim"])

    word_vectors = dict()
    pinyin_vectors = dict()
    exist_word_nums = 0
    exist_pinyin_nums = 0
    for (text_list, pinyin_list, _) in data_lists:
        # word vectors
        for word in text_list:
            if pre_train_word_vector_dicts:
                if word not in word_vectors:
                    if word in pre_train_word_vector_dicts:
                        word_vectors[word] = pre_train_word_vector_dicts[word]
                        exist_word_nums += 1
                    else:
                        word_vectors[word] = np.random.uniform(low=-0.5, high=0.5, size=word_vector_dim)
            else:
                if word not in word_vectors:
                    word_vectors[word] = np.random.uniform(low=-0.5, high=0.5, size=word_vector_dim)
        # pinyin vectors
        for pinyin in pinyin_list:
            if pre_train_pinyin_vector_dicts:
                if pinyin not in pinyin_vectors:
                    if pinyin in pre_train_pinyin_vector_dicts:
                        pinyin_vectors[pinyin] = pre_train_pinyin_vector_dicts[pinyin]
                        exist_pinyin_nums += 1
                    else:
                        pinyin_vectors[pinyin] = np.random.uniform(low=-0.5, high=0.5, size=pinyin_vector_dim)
            else:
                if pinyin not in pinyin_vectors:
                    pinyin_vectors[pinyin] = np.random.uniform(low=-0.5, high=0.5, size=pinyin_vector_dim)
    word_vectors[UNKNOWN] = np.random.uniform(low=-0.5, high=0.5, size=word_vector_dim)
    pinyin_vectors[UNKNOWN] = np.random.uniform(low=-0.5, high=0.5, size=pinyin_vector_dim)
    print(exist_word_nums, "/", len(word_vectors), " words find in pre trained word vectors")
    print(exist_pinyin_nums, "/", len(pinyin_vectors), " pinyin find in pre trained pinyin vectors")
    print("word vectors dict size:", len(word_vectors))
    print("pinyin vectors dict size:", len(pinyin_vectors))

    word_to_id_dicts = dict()
    pinyin_to_id_dicts = dict()
    word_vector_values = []
    pinyin_vector_values = []
    for idx, word in zip(range(0, len(word_vectors)), word_vectors.keys()):
        word_to_id_dicts[word] = idx
        word_vector_values.append(word_vectors[word])
    for idx, pinyin in zip(range(0, len(word_vectors)), pinyin_vectors.keys()):
        pinyin_to_id_dicts[pinyin] = idx
        pinyin_vector_values.append(pinyin_vectors[pinyin])
    return word_vector_values, pinyin_vector_values, word_to_id_dicts, pinyin_to_id_dicts


def generate_train_valid_data(configs, data_lists, word_to_id_dicts, pinyin_to_id_dicts, num_classes):
    x_data = []
    y_data = []
    word_max_len = int(configs["word_max_len"])
    pinyin_max_len = int(configs["pinyin_max_len"])
    unknown_word_id = word_to_id_dicts[UNKNOWN]
    unknown_pinyin_id = pinyin_to_id_dicts[UNKNOWN]
    for (text_list, pinyin_list, tag) in data_lists:
        x_item = []
        for word in text_list:
            if word in word_to_id_dicts:
                x_item.append(word_to_id_dicts[word])
            else:
                x_item.append(unknown_word_id)
        if len(x_item) > word_max_len:
            x_item = x_item[:word_max_len]
        else:
            while len(x_item) <= word_max_len:
                x_item.append(unknown_word_id)
        for pinyin in pinyin_list:
            if pinyin in pinyin_to_id_dicts:
                x_item.append(pinyin_to_id_dicts[pinyin])
            else:
                x_item.append(unknown_pinyin_id)
        if len(x_item) > word_max_len+pinyin_max_len:
            x_item = x_item[:word_max_len+pinyin_max_len]
        else:
            while len(x_item) <= word_max_len+pinyin_max_len:
                x_item.append(unknown_pinyin_id)
        x_data.append(x_item)
        y_score = [0.0]*num_classes
        y_score[tag] = 1.0
        y_data.append(y_score)
    valid_len = int(len(x_data)*configs["split"])
    x_valid_data = x_data[:valid_len]
    y_valid_data = y_data[:valid_len]
    x_train_data = x_data[valid_len:]
    y_train_data = y_data[valid_len:]
    print("train data nums:", len(x_train_data))
    print("valid data nums:", len(x_valid_data))
    return x_train_data, y_train_data, x_valid_data, y_valid_data


if __name__ == "__main__":
    # 解析配置json
    configs = json.load(open(os.path.join(ConfDirPath, "data_process_configs.json")))
    data_serialization_dir = os.path.join(DataDirPath, configs["serialization_dir"])
    if not os.path.exists(data_serialization_dir):
        os.mkdir(data_serialization_dir)
    # 读取数据集
    data_lists, file_to_tag_dict = load_data(configs)
    _save_vocab(file_to_tag_dict, os.path.join(data_serialization_dir, "file_to_tag.txt"))
    # 数据向量化
    word_vector_values, pinyin_vector_values, word_to_id_dicts, pinyin_to_id_dicts = data_vectorization(configs, data_lists)
    pickle.dump(np.asarray(word_vector_values, dtype=np.float32), open(os.path.join(data_serialization_dir, "word_vectors"), "wb"))
    pickle.dump(np.asarray(pinyin_vector_values, dtype=np.float32), open(os.path.join(data_serialization_dir, "pinyin_vectors"), "wb"))
    pickle.dump(word_to_id_dicts, open(os.path.join(data_serialization_dir, "word_to_id_dicts"), "wb"))
    pickle.dump(pinyin_to_id_dicts, open(os.path.join(data_serialization_dir, "pinyin_to_id_dicts"), "wb"))
    # 生成train和valid data
    x_train_data, y_train_data, x_valid_data, y_valid_data = generate_train_valid_data(configs, data_lists, word_to_id_dicts, pinyin_to_id_dicts, len(file_to_tag_dict))
    pickle.dump(x_train_data, open(os.path.join(data_serialization_dir, "x_train_data"), "wb"))
    pickle.dump(y_train_data, open(os.path.join(data_serialization_dir, "y_train_data"), "wb"))
    pickle.dump(x_valid_data, open(os.path.join(data_serialization_dir, "x_valid_data"), "wb"))
    pickle.dump(y_valid_data, open(os.path.join(data_serialization_dir, "y_valid_data"), "wb"))
