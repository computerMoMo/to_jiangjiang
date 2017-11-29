# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import codecs
import json
import pickle
import jieba

ConfDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "conf")
DataDirPath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
UNKNOWN = "<OOV>"


def get_data_ids(configs):
    test_reader = codecs.open(configs["input_file"], mode='r', encoding='utf-8')
    data_lists = []
    for line in test_reader.readlines():
        line_list = line.strip().split("\t")
        if configs["cut_word"]:
            text_item = jieba.lcut(line_list[0])
        else:
            text_item = [ch for ch in line_list[0]]
        pinyin_item = line_list[1].split(" ")
        data_lists.append((text_item, pinyin_item))

    data_id_lists = []
    word_unknown_id = word_to_id_dict[UNKNOWN]
    pinyin_unknown_id = pinyin_to_id_dict[UNKNOWN]
    word_max_len = configs["word_max_len"]
    pinyin_max_len = configs["pinyin_max_len"]
    for (text_list, pinyin_list) in data_lists:
        data_item = []
        for word in text_list:
            if word in word_to_id_dict:
                data_item.append(word_to_id_dict[word])
            else:
                data_item.append(word_unknown_id)
        if len(data_item) > word_max_len:
            data_item = data_item[:word_max_len]
        else:
            while len(data_item) <= word_max_len:
                data_item.append(word_unknown_id)

        for pinyin in pinyin_list:
            if pinyin in pinyin_to_id_dict:
                data_item.append(pinyin_to_id_dict[pinyin])
            else:
                data_item.append(pinyin_unknown_id)
        if len(data_item) > word_max_len + pinyin_max_len:
            data_item = data_item[:word_max_len + pinyin_max_len]
        else:
            while len(data_item) <= word_max_len + pinyin_max_len:
                data_item.append(pinyin_unknown_id)
        data_id_lists.append(data_item)
    return data_id_lists


if __name__ == "__main__":
    configs = json.load(open(os.path.join(ConfDirPath, "interface.json")))
    serialization_dir = os.path.join(DataDirPath, configs["serialization_dir"])
    word_to_id_dict = pickle.load(open(os.path.join(serialization_dir, configs["word_to_id"]), 'rb'))
    pinyin_to_id_dict = pickle.load(open(os.path.join(serialization_dir, configs["pinyin_to_id"]), 'rb'))
    # get data id
    data_id_lists = get_data_ids(configs)
    print("data nums in test file:", len(data_id_lists))

    # load model
    checkpoint_file = tf.train.latest_checkpoint(configs["check_point_dir"])
    if checkpoint_file is None:
        print("Cannot find a valid checkpoint file!")
        exit(0)
    print("Using checkpoint file : {}".format(checkpoint_file))

    graph = tf.Graph()
    with graph.as_default(), tf.Session() as sess:
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_name = "text_classification/model_input:0"
        dropout_name = "text_classification/dropout_keep_prob:0"
        predict_name = "text_classification/output/predictions:0"
        features_name = "text_classification/dropout/dropout_feature/mul:0"

        input_tensor = graph.get_tensor_by_name(input_name)
        dropout_tensor = graph.get_tensor_by_name(dropout_name)
        predict_tensor = graph.get_tensor_by_name(predict_name)
        features_tensor = graph.get_tensor_by_name(features_name)

        # model run
        data_tags = []
        data_features = []
        for data_item in data_id_lists:
            data_item_array = np.asarray(data_item, dtype=np.int32).reshape((1, -1))
            item_tag, item_features = sess.run(fetches=[predict_tensor, features_tensor],
                                               feed_dict={input_tensor: data_item_array, dropout_tensor: 1.0})
            data_tags.append(item_tag[0])
            data_features.append(item_features[0])

        # save results
        tag_res_writer = codecs.open(os.path.join(configs["output_dir"], "tags.txt"), mode='w', encoding='utf-8')
        for tag in data_tags:
            tag_res_writer.write(str(tag)+"\n")
        tag_res_writer.close()

        pickle.dump(data_features, open(os.path.join(configs["output_dir"], "data_features"), 'wb'))

    print("model run over")