# -*- coding: UTF-8 -*-
import re
import tensorflow as tf
from bert import modeling
import datetime as dt
import create_input
import tokenization
import numpy as np
import io_bert
import copy
import matplotlib.pyplot as plt
import bert.optimization as opt
import configparser as cp


def var(x):
    x = tf.cast(x, tf.float64)
    mean_x = tf.reduce_mean(x)
    sub_x = x - mean_x

    variance = tf.reduce_mean(sub_x ** 2)

    return variance


def cov(x, y=None):
    """劣质TensorFlow实现，用法见numpy"""
    x = tf.transpose(tf.cast(x, tf.float64))
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    sub_x = x - mean_x

    cov_xx = tf.matmul(tf.transpose(sub_x), sub_x) / tf.cast(tf.shape(x)[0] - 1, tf.float64)

    if y is None:
        return cov_xx
    else:
        y = tf.transpose(tf.cast(y, tf.float64))
    mean_y = tf.reduce_mean(y, axis=0, keepdims=True)
    sub_y = y - mean_y

    cov_yy = tf.matmul(tf.transpose(sub_y), sub_y) / tf.cast(tf.shape(y)[0] - 1, tf.float64)
    cov_xy = tf.matmul(tf.transpose(sub_x), sub_y) / tf.cast(tf.shape(y)[0] - 1, tf.float64)

    cov_mat = tf.concat([tf.concat([cov_xx, cov_xy], -1), tf.concat([cov_xy, cov_yy], -1)], 0)
    #return [[cov_xx, cov_xy], [cov_xy, cov_yy]]
    return cov_mat


def pearson(x, y):
    return tf.divide(cov(x, y), tf.sqrt(cov(x, x)) * tf.sqrt(cov(y, y)))


if __name__ == '__main__':

    config_reader = open("config.txt")
    config = config_reader.read().replace(" ", "")
    config_reader.close()

    start = dt.datetime.now()
    bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
    vocab_file = "chinese_L-12_H-768_A-12/vocab.txt"
    batch_size = int(re.findall(r"\d+", re.findall(r"batch_size=\d+", config)[0])[0])
    num_labels = 2
    max_seq_length = int(re.findall(r"\d+", re.findall(r"max_seq_length=\d+", config)[0])[0])

    iter_num = int(re.findall(r"\d+", re.findall(r"iter_num=\d+", config)[0])[0])  # epoch ≈ 8
    lr = float(re.findall(r"\d+.?\d*", re.findall(r"lr=\d+\.?\d*", config)[0])[0])  # 0.00002

    train_files = str(re.findall(r"[\w/.,]+", re.findall(r"train_files=[\w/.,]+", config)[0])[1]).split(",")
    test_files = str(re.findall(r"[\w/.,]+", re.findall(r"test_files=[\w/.,]+", config)[0])[1]).split(",")
    verify_files = str(re.findall(r"[\w/.,]+", re.findall(r"verify_files=[\w/.,]+", config)[0])[1]).split(",")

    # 记录超参数和结果
    log = list()
    log.append("batch size: " + str(batch_size) + "\n")
    log.append("max sentence length: " + str(max_seq_length) + "\n")
    log.append("iteration times: " + str(iter_num) + "\n")
    log.append("learning rate: " + str(lr) + "\n")
    log.append("train files: " + str(train_files) + "\n")
    log.append("test files: " + str(test_files) + "\n")
    log.append("verify files: " + str(verify_files) + "\n")

    init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"

    is_training = True

    # 获取模型中所有的训练参数。
    tvars = tf.trainable_variables()

    # 加载BERT模型
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    if max_seq_length > bert_config.max_position_embeddings:  # 模型有个最大的输入长度 512
        raise ValueError("超出模型最大长度")

    #  创建bert的输入
    input_ids = tf.placeholder(shape=[batch_size, max_seq_length], dtype=tf.int32)
    input_mask = tf.placeholder(shape=[batch_size, max_seq_length], dtype=tf.int32)
    segment_ids = tf.placeholder(shape=[batch_size, max_seq_length], dtype=tf.int32)
    input_labels = tf.placeholder(shape=batch_size, dtype=tf.int32)

    # 创建bert模型
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )

    output_layer = model.get_pooled_output()
    init_output_layer = copy.copy(output_layer)

    hidden_size = output_layer.shape[-1].value  # 获取输出的维度

    # output
    output_weights = tf.Variable(tf.truncated_normal([num_labels, hidden_size], stddev=0.2))
    output_bias = tf.Variable(tf.zeros([num_labels]))

    output_linear = tf.add(tf.matmul(output_layer, output_weights, transpose_b=True), output_bias)

    with tf.variable_scope("loss"):

        if is_training:
            # 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, rate=0.1)

        log_probs = tf.nn.log_softmax(output_linear, axis=-1)
        one_hot_labels = tf.one_hot(input_labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)

        d = tf.cast(tf.shape(output_layer)[1], tf.float64)
        WBCAA_loss = tf.cast(tf.reduce_sum(
            tf.pow(tf.subtract(
                cov(init_output_layer),
                cov(output_layer)), 2.)) /
                             4*tf.square(d), tf.float32)  # 4 * depth^2

        loss = tf.reduce_mean(per_example_loss) + WBCAA_loss / 100   # λ = 1e-2
        predict = tf.argmax(tf.nn.softmax(output_linear), axis=1)  # argmax不可导
        acc = tf.reduce_mean(tf.cast(tf.equal(input_labels, tf.cast(predict, dtype=tf.int32)), "float"))

        texts, texts2, labels = io_bert.read_10000(train_files)
        train_step = iter_num * int(len(labels) / batch_size) - iter_num

        #train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        train_op = opt.create_optimizer(loss, lr, train_step, 100, False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        losses = []
        accuracies = []

        # train
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
        input_idsList = []
        input_masksList = []
        segment_idsList = []

        for t, t2, l in zip(texts, texts2, labels):
            single_input_id, single_input_mask, single_segment_id = create_input.convert_single_example(max_seq_length,
                                                                                                       tokenizer, t, t2)
            input_idsList.append(single_input_id)
            input_masksList.append(single_input_mask)
            segment_idsList.append(single_segment_id)

        input_idsList = np.asarray(input_idsList, dtype=np.int32)
        input_masksList = np.asarray(input_masksList, dtype=np.int32)
        segment_idsList = np.asarray(segment_idsList, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)

        num = 0
        for j in range(iter_num):
            #for i in range(iter_num):
            for i in range(int(labels.size/batch_size) - 1):
                num += 1
                continuousIndex = np.arange(i * batch_size, i * batch_size + batch_size)
                #shuffIndex = np.random.permutation(np.arange(len(texts)))[:batch_size]
                batch_labels = labels[continuousIndex]
                batch_input_idsList = input_idsList[continuousIndex]
                batch_input_masksList = input_masksList[continuousIndex]
                batch_segment_idsList = segment_idsList[continuousIndex]
                l, a, _, wbcaa = sess.run([loss, acc, train_op, WBCAA_loss], feed_dict={
                    input_ids: batch_input_idsList, input_mask: batch_input_masksList,
                    segment_ids: batch_segment_idsList, input_labels: batch_labels
                })
                losses.append(l)
                accuracies.append(a)
                print("train准确率:{},损失函数:{},进度：{}/{}".format(a, l, num, iter_num*int(labels.size/batch_size) - iter_num))

        print("train平均准确率:{},损失函数:{}".format(np.mean(accuracies), np.mean(losses)))
        log.append("train accuracy: " + str(np.mean(accuracies)) + "\t" + "train loss: " + str(np.mean(losses)) + "\n")
        plt.plot(losses)
        # test
        texts, texts2, labels = io_bert.read_10000(test_files)

        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
        input_idsList = []
        input_masksList = []
        segment_idsList = []

        for t, t2, l in zip(texts, texts2, labels):
            single_input_id, single_input_mask, single_segment_id = create_input.convert_single_example(max_seq_length,
                                                                                                                    tokenizer, t, t2)
            input_idsList.append(single_input_id)
            input_masksList.append(single_input_mask)
            segment_idsList.append(single_segment_id)

        input_idsList = np.asarray(input_idsList, dtype=np.int32)
        input_masksList = np.asarray(input_masksList, dtype=np.int32)
        segment_idsList = np.asarray(segment_idsList, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)

        losses = []
        accuracies = []
        TP = 0
        FP = 0

        T = 0

        for i in range(int(labels.size/batch_size)):
            continuousIndex = np.arange(i*batch_size, i*batch_size + batch_size)
            batch_labels = labels[continuousIndex]
            batch_input_idsList = input_idsList[continuousIndex]
            batch_input_masksList = input_masksList[continuousIndex]
            batch_segment_idsList = segment_idsList[continuousIndex]
            l, a, p = sess.run([loss, acc, predict], feed_dict={
                input_ids: batch_input_idsList, input_mask: batch_input_masksList,
                segment_ids: batch_segment_idsList, input_labels: batch_labels
            })
            accuracies.append(a)
            losses.append(l)

            p = [int(val) for val in p]

            for label, pred, index in zip(batch_labels, p, continuousIndex):
                if label == 0:
                    T += 1
                    if pred == 0:
                        TP += 1
                if label == 1:
                    if pred != label:
                        FP += 1

        accuracy = np.mean(accuracies)

        recall = "NaN"
        precision = "NaN"
        formula_1 = "NaN"
        if TP * T != 0:
            recall = TP / T
            precision = TP / (TP + FP)

            formula_1 = 2 * recall * precision / (recall + precision)

        print("test最终准确率:{},损失函数:{},F1:{}:".format(accuracy, np.mean(losses), formula_1))
        log.append("test accuracy: " + str(np.mean(accuracies)) + "\t" + "test loss: " + str(np.mean(losses)) + "\n")
        log.append("F1: " + str(formula_1) + "\t" + "recall: " + str(recall) + "\t" + "precision: " + str(precision) + "\n")

        """with open("data/data_split/2_spy.txt", "w") as w:
            w.writelines(spies)"""
        """"# verify
        texts, texts2, labels = iotest.read_10000(verify_files, max_seq_length)
    
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
        input_idsList = []
        input_masksList = []
        segment_idsList = []
    
        for t, t2 in zip(texts, texts2):
            single_input_id, single_input_mask, single_segment_id = create_input.convert_single_example(max_seq_length,
                                                                                                        tokenizer, t, t2)
            input_idsList.append(single_input_id)
            input_masksList.append(single_input_mask)
            segment_idsList.append(single_segment_id)
    
        input_idsList = np.asarray(input_idsList, dtype=np.int32)
        input_masksList = np.asarray(input_masksList, dtype=np.int32)
        segment_idsList = np.asarray(segment_idsList, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
    
        losses = []
        accuracies = []
        for i in range(int(labels.size/batch_size)):
            continuousIndex = np.arange(i*batch_size, i*batch_size + batch_size)
            batch_labels = labels[continuousIndex]
            batch_input_idsList = input_idsList[continuousIndex]
            batch_input_masksList = input_masksList[continuousIndex]
            batch_segment_idsList = segment_idsList[continuousIndex]
            l, a = sess.run([loss, acc], feed_dict={
                input_ids: batch_input_idsList, input_mask: batch_input_masksList,
                segment_ids: batch_segment_idsList, input_labels: batch_labels
            })
            accuracies.append(a)
            losses.append(l)
        print("verify最终准确率:{},损失函数:{}".format(np.mean(accuracies), np.mean(losses)))
        log.append("verify accuracy: " + str(np.mean(accuracies)) + "\t" + "verify loss: " + str(np.mean(losses)) + "\n")"""
    end = dt.datetime.now()

    log.append("time duration: " +
               str(int((end - start).total_seconds()/60)) + ":" +
               str(int((end - start).total_seconds()) - int((end - start).total_seconds()/60) * 60) + "\n")
    log.append("using experiment dataset with coral loss" + "\n")

    # 序列化

    end = end.strftime("%Y%m%d-%H-%M-%S")
    with open("logs/{}.txt".format(end), "w", encoding="UTF-8") as writer:
        writer.writelines(log)

    plt.waitforbuttonpress()
