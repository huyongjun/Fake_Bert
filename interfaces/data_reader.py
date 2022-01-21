import pymongo
import interfaces.date_gen
import pygraphviz as viz

DB_HOST = "192.168.8.161"


def get_comment_data_by_date(date_start: str, date_end: str = None, limit_per_day=500):
    """来自third_party.py"""
    import datetime
    if date_end is None:
        date_list = [date_start]
    else:
        date_list = interfaces.date_gen.get_date_list(date_start, date_end)
    myclient = pymongo.MongoClient("mongodb://" + DB_HOST + ":27017/")
    mydb = myclient["weibo"]
    tweet_col = mydb["口罩评论_优化_固定"]
    for date in date_list:
        date_str = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m %d')
        date_cn = date_str.split()[0].lstrip("0") + "月" + date_str.split()[1].lstrip("0") + "日"
        tweet_query = {"时间": {"$regex": "{}.*".format(date_cn)}}
        tweet_doc = tweet_col.find(tweet_query).sort([("点赞", -1)]).limit(limit_per_day)
        for tweet in tweet_doc:
            yield tweet


def get_comment_data_by_blog(blog_ids):
    """来自third_party.py，调试中。。。"""
    myclient = pymongo.MongoClient("mongodb://" + DB_HOST + ":27017/")
    mydb = myclient["weibo"]
    tweet_col = mydb["评论测试_优化_固定"]

    if type(blog_ids) is str:
        blog_ids = [blog_ids]

    i = 0
    for blog_id in blog_ids:
        i += 1
        tweet_query = {"博文_id": {"$regex": blog_id}}
        tweet_doc = tweet_col.find(tweet_query)
        print(str(tweet_query) + "searching" + "({}/{})".format(i, len(blog_ids)))
        for tweet in tweet_doc:
            yield tweet


def get_hot_search_data_by_blog(blog_ids):
    """
        利用博客编号搜素数据

        :param *blog_ids 博客编号（eg. Iu8YobgCI）

        """
    myclient = pymongo.MongoClient("mongodb://"+DB_HOST+":27017/")
    mydb = myclient["weibo"]
    tweet_col = mydb["热搜博文"]

    if type(blog_ids) is str:
        blog_ids = [blog_ids]
    for blog_id in blog_ids:
        tweet_query = {"博文_id": {"$regex": blog_id}}
        tweet_doc = tweet_col.find(tweet_query)
        for tweet in tweet_doc:
            # to_return = {
            #     "heading": tweet["标题"],
            #     "content": tweet["内容"],
            #     "topic_url": tweet["内容_链接"],
            #     "url": tweet["内容_链接9"],
            #     "date": date,
            #     "share": tweet["cardact"],
            #     "comment": tweet["cardact14"],
            #     "like": tweet["cardact15"]
            # }
            yield tweet


def get_hot_search_data_by_blogs(blog_ids):
    """
        利用博客编号搜素数据2.0，注意返回的是数组而不是迭代器

        :param *blog_ids 博客编号（eg. Iu8YobgCI）
        """
    myclient = pymongo.MongoClient("mongodb://"+DB_HOST+":27017/")
    mydb = myclient["weibo"]
    tweet_col = mydb["热搜博文"]

    if type(blog_ids) is str:
        blog_ids = [blog_ids]
    tweet_query = {"博文_id": {"$in": blog_ids}}
    tweet_doc = tweet_col.find(tweet_query)
    return tweet_doc


def get_hot_search_data_by_date(date_start: str, date_end: str = None):
    """
    利用日期搜素数据

    :param date_start 起始日期
    :param date_end 结束日期，若留空则仅起始日期

    """
    import datetime
    if date_end is None:
        date_list = [date_start]
    else:
        date_list = interfaces.date_gen.get_date_list(date_start, date_end)
    myclient = pymongo.MongoClient("mongodb://"+DB_HOST+":27017/")
    mydb = myclient["weibo"]
    tweet_col = mydb["热搜博文"]
    for date in date_list:
        date_str = datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%m %d')
        date_cn = date_str.split()[0]+"月"+date_str.split()[1]+"日"
        tweet_query = {"时间": {"$regex": date_cn}}
        tweet_doc = tweet_col.find(tweet_query)
        for tweet in tweet_doc:
            # to_return = {
            #     "heading": tweet["标题"],
            #     "content": tweet["内容"],
            #     "topic_url": tweet["内容_链接"],
            #     "url": tweet["内容_链接9"],
            #     "date": date,
            #     "share": tweet["cardact"],
            #     "comment": tweet["cardact14"],
            #     "like": tweet["cardact15"]
            # }
            yield tweet

if __name__ == '__main__':
    G = viz.AGraph()
    G.add_node('a')
    G.add_edge('b', 'c')
    G.add_edge('b', 'd')
    G.layout()
    G.draw('file.png')

