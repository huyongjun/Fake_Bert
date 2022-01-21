import datetime

def get_date_list(date_start:str,date_end:str):
	# 创建日期辅助表

	# if datestart is None:
	# 	datestart = '2016-01-01'
	# if dateend is None:
	# 	dateend = datetime.datetime.now().strftime('%Y-%m-%d')

	# 转为日期格式
	date_start=datetime.datetime.strptime(date_start,'%Y-%m-%d')
	date_end=datetime.datetime.strptime(date_end,'%Y-%m-%d')
	date_list = []
	# date_list.append(date_start.strftime('%Y-%m-%d'))
	while date_start<=date_end:
        # 日期转字符串存入列表
		date_list.append(date_start.strftime('%Y-%m-%d'))
		# 日期叠加一天
		date_start+=datetime.timedelta(days=+1)
	return date_list


def get_date_list_chinese(date_start:str,date_end:str):
	# 创建日期辅助表

	# if datestart is None:
	# 	datestart = '2016-01-01'
	# if dateend is None:
	# 	dateend = datetime.datetime.now().strftime('%Y-%m-%d')

	# 转为日期格式
	date_start=datetime.datetime.strptime(date_start,'%Y-%m-%d')
	date_end=datetime.datetime.strptime(date_end,'%Y-%m-%d')
	date_list = []
	# date_list.append(date_start.strftime('%Y-%m-%d'))
	while date_start<date_end:
        # 日期转字符串存入列表
		date_str = date_start.strftime('%m %d')
		date_list.append(date_str.split()[0]+"月"+date_str.split()[1]+"日")
		# 日期叠加一天
		date_start+=datetime.timedelta(days=+1)
	return date_list