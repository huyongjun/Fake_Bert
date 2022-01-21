# -*- coding: UTF-8 -*-
 
import httplib2
import hashlib
import urllib.parse
import random
import json

def translate(q):
    appid = '20190820000328034'  # 你的appid
    secretKey = 'POwV1lZO9f3cQtgFgGql'  # 你的密钥

    myurl = '/api/trans/vip/translate'

    fromLang = 'zh'
    toLang = 'en'
    salt = random.randint(32768, 65536)

    httpClient = None
    sign = appid+q+str(salt)+secretKey
    m1 = hashlib.md5()
    m1.update(sign.encode("UTF-8"))
    sign = m1.hexdigest()
    myurl = myurl+'?appid='+appid+'&q='+urllib.parse.quote(q)+'&from='+fromLang+'&to='+toLang+'&salt='+str(salt)+'&sign='+sign

    try:
        httpClient = httplib2.HTTPConnectionWithTimeout('api.fanyi.baidu.com', timeout=15)
        httpClient.request('GET', myurl)

        #response是HTTPResponse对象
        response = httpClient.getresponse()
        response = response.read().decode("UTF-8")
        json_file = json.loads(response)

        src = json_file["trans_result"][0]["src"]
        dst = json_file["trans_result"][0]["dst"]
    except:
        return "nan"
    finally:
        if httpClient:
            httpClient.close()

    return src, dst
