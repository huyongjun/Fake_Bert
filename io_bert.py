import os
import numpy as np
import datetime as time
import re
import matplotlib.pyplot as plt
import enum
# pd.read_csv("data/gold_standard/true/fold1/t_hilton_1.txt")
import interfaces.data_reader as dr

t = "\t"


def read_100k(file_names):
    """
    :param file_names: 要读取的文件名(data/file_names.txt)(仅限训练用的weibo_senti_100k)

    :return: data: 评论内容
             label:标签 (假为0)
    """
    data = []
    label = []

    if isinstance(file_names, str):  # 如果只有一个str
        with open("data/{}".format(file_names), encoding="UTF-8") as reader:
            for line in reader.read().splitlines():
                line = line.split(",", 1)
                data.append(line[1])
                label.append(line[0])
    else:  # 如果是一个数组
        for file_name in file_names:
            with open("data/{}.txt".format(file_name), encoding="UTF-8") as reader:
                for line in reader.read().splitlines():
                    line = line.split("\t", 1)
                    data.append(line[1])
                    label.append(line[0])

    return data, label


def read_comment(file_names):
    """
    :param file_names: 要读取的文件名(data/file_names)(仅限自己爬取的格式)

    :return: data: 评论
    """

    prefix = []
    data = []
    appendix = []

    if isinstance(file_names, str):  # 如果只有一个str
        with open("data/{}".format(file_names), encoding="UTF-8") as reader:
            for line in reader.read().splitlines():
                line = line.split(",")
                if len(line) == 7:
                    prefix.append(",".join(line[0:3]))
                    data.append(line[4])
                    appendix.append(",".join(line[5:]))
    else:  # 如果是一个数组
        for file_name in file_names:
            with open("data/{}".format(file_name), encoding="UTF-8") as reader:
                for line in reader.read().splitlines():
                    line = line.split(",")
                    prefix.append(",".join(line[0:3]))
                    data.append(line[4])
                    appendix.append(",".join(line[4:]))

    return prefix, data, appendix


def emo():
    emo_dictionary = {}
    with open("data/Emo.txt", encoding="UTF-8") as reader:
        lines = reader.readlines()
        for line in lines:
            line = line.split("\t")
            emo_dictionary[line[0]] = int(line[2])

    data, label = read_100k("data_simplified/train_experiment")

    emo_values = []

    for d in data:
        emo_value = 0
        for key, value in emo_dictionary.items():
            emo_value += len(re.findall(key, d)) * value
        emo_values.append(emo_value)

    with open("data/data_simplified/train_emo.txt", "w") as writer:
        for l, d, e in zip(label, data,  emo_values):
            writer.write(str(l) + "\t" + d + "\t" + str(e) + "\n")


def regroup(test_size, verify_size, file_name):
    train_size = 1 - test_size - verify_size
    data = []

    with open("data/{}".format(file_name), encoding="UTF-8") as reader:
        for line in reader.read().splitlines():
            line = line.split(",", 1)
            data.append(line)

    train = data[0: int(train_size*len(data))]
    test = data[int(train_size*len(data)+1): int(train_size*len(data)+1+test_size*len(data))]
    verify = data[int(-verify_size*len(data)): -1]

    return train, test, verify


def read_10000_with_emo(file_names):
    """
    :param file_names: 要读取的文件名(data/file_names.txt)

    :return: data: 评论内容
             data2: 购买时间 + “ ” + 评论时间 + “ ” + 用户名
             label:标签 (假为0)
    """
    data = []
    label = []
    emo = []

    if isinstance(file_names, str):  # 如果只有一个str
        with open("data/{}.txt".format(file_names), encoding="UTF-8") as reader:
            for line in reader.read().splitlines():
                line = line.split("\t")
                data.append(line[1])
                label.append(line[0])
                emo.append(line[5])
    else:  # 如果是一个数组
        for file_name in file_names:
            with open("data/{}.txt".format(file_name), encoding="UTF-8") as reader:
                for line in reader.read().splitlines():
                    line = line.split("\t")
                    data.append(line[1])
                    label.append(line[0])
                    emo.append(line[5])

    return data, label, emo


def coefficient_of_var(x):
    return np.sqrt(np.var(x)) / np.mean(x)


def read(file_names):
    """
    :param file_names: 要读取的文件名(data/file_names)(仅限博文)

    :return: data
    """

    prefix = []
    data = []
    appendix = []

    if isinstance(file_names, str):  # 如果只有一个str
        with open("data/{}".format(file_names), encoding="UTF-8") as reader:
            for line in reader.read().splitlines():
                line = line.split(",")
                if len(line) == 7:
                    prefix.append(",".join(line[0:6]))
                    data.append(line[6])
                    appendix.append(",".join(line[7:]))
                else:
                    print("err")

    return prefix, data, appendix


class DataSearchMethod(enum.Enum):
    blog_date = 0
    blog_id = 1
    comment_date = 3
    comment_id = 4


def read_from_database(type:DataSearchMethod, *args):
    """
    :param type 要使用的数据（评论或博文）以及要使用的搜索方法（日期或博文编号）,
                其余要传入的参数请参考interfaces/data_reader


    :return: data
    """

    id = []
    data = []

    collected = []

    db = []

    if type is DataSearchMethod.blog_date:
        temp = dr.get_hot_search_data_by_date(args[0], args[1])
        for blog in temp:
            db.append(blog)
    elif type is DataSearchMethod.blog_id:
        db = dr.get_hot_search_data_by_blogs(args[0])

    elif type is DataSearchMethod.comment_date:
        temp = dr.get_comment_data_by_date(args[0], args[1], args[2])
        for blog in temp:
            db.append(blog)
    else:
        temp = dr.get_comment_data_by_blog(args[0])
        for blog in temp:
            db.append(blog)

    for each in db:
        id.append(each["_id"])
        data.append(each["内容"])

    return id, data


if __name__ == '__main__':

    """train, test, verify = regroup(0.01, 0.01, "weibo_senti_100k.csv")

    with open("data/train.csv", "w", encoding="UTF_8") as writer:
        for l, d in train:
            writer.write(str(l) + "," + str(d) + "\n")

    with open("data/test.csv", "w", encoding="UTF_8") as writer:
        for l, d in test:
            writer.write(str(l) + "," + str(d) + "\n")

    with open("data/verify.csv", "w", encoding="UTF_8") as writer:
        for l, d in verify:
            writer.write(str(l) + "," + str(d) + "\n")"""

    _, data, _ = read_comment("Comments.csv")

    for datum in data:
        print(datum)

