# -*- coding: utf-8 -*-
import pandas as pd
import urllib.request as req
import json
import sys
import time
import random
import os

print(sys.getdefaultencoding())


class JDCommentsCrawler:

    def __init__(self, productId=None, callback=None, page=1, score=0, sortType=5, pageSize=15):
        self.productId = productId  # 商品ID
        self.score = score  # 评论类型（好：3、中：2、差：1、所有：0）
        self.sortType = sortType  # 排序类型（推荐：5、时间：6）
        self.pageSize = pageSize  # 每页显示多少条记录（默认10）
        self.callback = callback  # 回调函数，每个商品都不一样
        self.page = page
        self.locationLink = 'https://sclub.jd.com/comment/productPageComments.action'
        self.paramValue = {
            'callback': self.callback,
            'productId': self.productId,
            'score': self.score,
            'sortType': self.sortType,
            'pageSize': self.pageSize,
        }
        self.locationUrl = None

    def paramDict2Str(self, params):
        str1 = ''
        for p, v in params.items():
            str1 = str1 + p + '=' + str(v) + '&'
        return str1

    def concatLinkParam(self):
        self.locationUrl = self.locationLink + '?' + self.paramDict2Str(self.paramValue) + 'isShadowSku=0&fold=1&page=0'
        # print(self.locationUrl)

    def requestMethod(self):
        headers = {
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0',
            'Referer': 'https://item.jd.com/%d.html' % (self.productId),
            'Host': 'sclub.jd.com'
        }
        reqs = req.Request(self.locationUrl, headers=headers)
        print('reqs : ', reqs)
        return reqs

    def showList(self):
        request_m = self.requestMethod()
        conn = req.urlopen(request_m)
        return_str = conn.read().decode('gbk')
        return_str = return_str[len(self.callback) + 1:-2]
        return json.loads(return_str)

    def requestMethodPage(self, p):
        # 伪装浏览器 ，打开网站
        headers = {
            'Connection': 'Keep-Alive',
            'Accept': 'text/html, application/xhtml+xml, */*',
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0',
            'Referer': 'https://item.jd.com/%d.html' % (self.productId),
            'Host': 'sclub.jd.com'
        }
        url = self.locationUrl[:-1] + str(p)
        print('url : ', url)
        reqs = req.Request(url, headers=headers)
        return reqs

    def showListPage(self, p):
        request_m = self.requestMethodPage(p)
        conn = req.urlopen(request_m)
        return_str = conn.read().decode('gbk')
        return_str = return_str[len(self.callback) + 1:-2]
        return json.loads(return_str)

    def save_csv(self, df, p):
        # 保存文件
        df.to_csv(path_or_buf=os.getcwd()+'\\temp\\jd_%d.txt' % p, encoding='utf-8')

    def crawler(self):
        # 把抓取的数据存入CSV文件，设置时间间隔，以免被屏蔽
        dfs = []
        for p in range(self.page):
            json_info = self.showListPage(p)
            tmp_list = []
            # print(json_info)
            productCommentSummary = json_info['productCommentSummary']
            productId = productCommentSummary['productId']
            comments = json_info['comments']
            for com in comments:
                tmp_list.append([com['id'], com['content'].replace("\n", " "), com['creationTime'], com['referenceTime'],
                                 com['nickname']])
            df = pd.DataFrame(tmp_list, columns=['comment_id', 'content', 'create_time', 'reference_time',
                                                'nickname'])
            self.save_csv(df, p)
            dfs.append(df)
            time.sleep(random.randint(4,5))
        final_df = pd.concat(dfs, ignore_index=True)
        self.save_csv(final_df, self.page)


def jdComment(page, productId):  # 某iphone手机
    # 设置关键变量
    callback = 'fetchJSON_comment98vv782'  # 回调函数
    JDC = JDCommentsCrawler(productId, callback, page)
    JDC.concatLinkParam()
    JDC.crawler()


if __name__ == '__main__':
    jdComment(10, 5089273)
