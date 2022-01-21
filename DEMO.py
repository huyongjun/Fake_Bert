import tensorflow as tf
from bert import modeling
import datetime as dt
import create_input
import tokenization
import numpy as np
import io_bert
import configparser as cp


if __name__ == '__main__':

    start = dt.datetime.now()

    """texts = io_bert.read_comment("Comments.csv")"""

    blogs = []

    with open("C:/Users/Administrator/Desktop/Yishan/qiuchengtong（超网络）/SNA/data/mask_id.txt", encoding="UTF-8") as reader:
        for line in reader.readlines():
            blogs.append(line[:-1])

    ids, data = io_bert.read_from_database(io_bert.DataSearchMethod.comment_date, "2020-01-19", "2020-09-11", 30000)

    print("处理完毕，共获得{}条评论数据".format(len(ids)))

    cfg = cp.ConfigParser()
    cfg.read("config.ini")

    batch_size = cfg.getint("hyperParam", "batch_size")
    max_seq_length = cfg.getint("hyperParam", "max_seq_length")

    bert_config = modeling.BertConfig.from_json_file("chinese_L-12_H-768_A-12/bert_config.json")
    vocab_file = "chinese_L-12_H-768_A-12/vocab.txt"

    num_labels = 2

    init_checkpoint = "chinese_L-12_H-768_A-12/bert_model.ckpt"

    is_training = False

    # 获取模型中所有的训练参数。
    tvars = tf.trainable_variables()

    # 加载BERT模型
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    if max_seq_length > bert_config.max_position_embeddings:
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

    hidden_size = output_layer.shape[-1].value  # 获取输出的维度

    # output
    output_weights = tf.Variable(tf.truncated_normal([num_labels, hidden_size], stddev=0.2))
    output_bias = tf.Variable(tf.zeros([num_labels]))

    output_linear = tf.add(tf.matmul(output_layer, output_weights, transpose_b=True), output_bias)

    with tf.variable_scope("loss"):
        predict = tf.nn.softmax(output_linear)

    print("读取模型中")
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        new_saver = tf.train.Saver()
        new_saver.restore(sess, tf.train.latest_checkpoint('saved_models2'))

        print("读取完毕，进行预测中")
        tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file)
        input_idsList = []
        input_masksList = []
        segment_idsList = []

        for t in data:
            single_input_id, single_input_mask, single_segment_id = create_input.convert_single_example(max_seq_length,
                                                                                                        tokenizer, t)
            input_idsList.append(single_input_id)
            input_masksList.append(single_input_mask)
            segment_idsList.append(single_segment_id)

        input_idsList = np.asarray(input_idsList, dtype=np.int32)
        input_masksList = np.asarray(input_masksList, dtype=np.int32)
        segment_idsList = np.asarray(segment_idsList, dtype=np.int32)

        T = 0
        F = 0
        p = []

        for i in range(int(input_idsList.shape[0]/batch_size)):
            continuousIndex = np.arange(i*batch_size, i*batch_size + batch_size)
            batch_input_idsList = input_idsList[continuousIndex]
            batch_input_masksList = input_masksList[continuousIndex]
            batch_segment_idsList = segment_idsList[continuousIndex]

            p0 = sess.run([predict], feed_dict={
                input_ids: batch_input_idsList, input_mask: batch_input_masksList,
                segment_ids: batch_segment_idsList})

            p.extend(p0[0])

        out = []

        with open("data/emotion/start_to_2020_09_31_comment.txt", mode="w", encoding="UTF-8") as writer:
            for id, p0 in zip(ids, p):
                writer.write(",".join([str(id), str(p0[0])])+"\n")

    end = dt.datetime.now()

    print("耗时: " +
          str(int((end - start).total_seconds()/60)) + ":" +
          str(int((end - start).total_seconds()) - int((end - start).total_seconds()/60) * 60) + "\n")
