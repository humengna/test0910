#coding:gbk
"""
A股股票策略 [回测版 - 提速版]：热门概念池 + 对数线性回归动量打分 + RSRS修正标准分大盘择时
                              + 动量分数连续下降个股择时 + 固定-15%硬止损

相对原回测版的提速点（交易逻辑保持一致）：
  1. 行情一次性预加载：首根有效 bar 时把全市场 close/open/low/preClose/suspendFlag
     一次性读入内存，拼成“日期 x 股票”的宽表(numpy)。之后每根 bar 只做数组切片，
     不再每天对 5000 只股票调用 get_market_data_ex（原来每根 bar 调 2 次全市场 + 若干次单股）。
  2. 静态信息缓存：板块成分股、get_instrument_detail（ST/市值）只查一次，
     原来每根 bar 对 5000 只股票逐只调用 get_instrument_detail，是最大的耗时点。
  3. 动量打分向量化：对全部股票一次矩阵运算完成回归（斜率 + R2），
     替代逐只 np.polyfit 的 Python 循环。
  4. RSRS 一次性计算：用 pandas rolling 向量化算出全部日期的 beta/R2/zscore，
     原来每根 bar 做 600 次 np.polyfit。
  5. 去掉每根 bar 打印全部 sorted_stocks（几千条）的日志，控制台输出本身非常慢。
     需要详细日志时把 VERBOSE 设为 True。

板块分开：
  主板 与 创业板+科创板 作为两个独立的组（见 BOARD_GROUPS），各自选第1名、各自择时、
  各自管理本板块持仓，资金按占比分配；只保留一组即可单独回测某个板块。
"""

import time
import pandas as pd
import numpy as np

# ============================================================
# 全局变量：保存跨 bar 的状态
# ============================================================
class G:
	pass

g = G()

# 策略名称
STRATEGY_NAME = '动量择时策略'

# 概念板块列表
CONCEPT_SECTORS = [
	'锂电池', '芯片', '人工智能', '光伏', '军工', '新能源车', '储能',
	'5G', '半导体', '国产软件', '云计算', '大数据', '物联网', '机器人',
	'氢能源', '风能', '核电', '特高压', '充电桩', '智能电网', '工业互联网',
	'数字货币', '区块链', '元宇宙', 'VR', '消费电子', '汽车电子', '无人驾驶',
	'高端装备', '新材料', '稀土永磁', '石墨烯', '碳纤维', '降解塑料',
	'医美', '创新药', '生物疫苗', '基因测序', '医疗器械', '中药',
	'白酒', '食品饮料', '免税', '电商', '网红经济', '在线教育',
	'卫星导航', '大飞机', '军民融合', '一带一路', '雄安新区', '海南自贸',
	'碳中和', '环保', '固废处理', '污水处理', '垃圾分类',
	'网络安全', '信创', '东数西算', '量子科技', '脑机接口',
]
CONCEPT_SECTORS = ['沪深a股']
# 过滤条件
MIN_MARKET_CAP = 30e8
MAX_MARKET_CAP = 500e8

# 动量打分参数
LOOKBACK_DAYS = 5   #29
TRADING_DAYS_PER_YEAR = 244

# RSRS 参数
RSRS_N = 21
RSRS_M = 600
RSRS_INDEX = '000300.SH'

# 止损线
STOP_LOSS_RATIO = -0.15

# 板块分组：每组独立选股、独立择时、独立持仓，值为该组资金占比
#   'main'     主板（沪市 60xxxx、深市 000/001/002/003）
#   'gem_star' 创业板（300/301）+ 科创板（688/689）
# 只回测某一个板块：只保留一项并设为 1.0，例如 BOARD_GROUPS = {'main': 1.0}
BOARD_GROUPS = {
	'main': 0.5,
	'gem_star': 0.5,
}
BOARD_NAMES = {'main': '主板', 'gem_star': '创业板+科创板'}

# 预加载时在回测起点之前多取的自然日（需覆盖 LOOKBACK_DAYS + 10 根以上的交易日）
PRELOAD_BUFFER_DAYS = 90

# 详细日志开关（打印全部排序结果等，会明显拖慢回测）
VERBOSE = False

PRICE_FIELDS = ['close', 'open', 'low', 'preClose', 'suspendFlag']


# ============================================================
# 系统函数
# ============================================================

def init(C):
	"""
	回测初始化
	"""
	g.account = "testS"	   # 回测模拟账号
	g.acct_type = "STOCK"	 # 股票账号
	g.stock_df = {}		   # 各板块组目标股的近5日动量分数序列
	g.today_target = {}	   # 各板块组今日目标股票
	g.bar_count = 0		   # 已处理 bar 计数
	g.loaded = False		  # 行情是否已预加载
	g.name_cache = {}		 # 股票名称缓存

	print('[动量择时策略-回测版] 初始化完成')
	print(f'  回测账号: {g.account}, 类型: {g.acct_type}')
	print(f'  概念板块数: {len(CONCEPT_SECTORS)}')
	print(f'  动量回看: {LOOKBACK_DAYS}天, 止损线: {STOP_LOSS_RATIO:.0%}')
	print(f'  板块分组: {[(BOARD_NAMES.get(k, k), w) for k, w in BOARD_GROUPS.items()]}')


def handlebar(C):
	"""
	每根日K线触发一次，模拟完整交易日流程：
	  ① 选股 + 动量打分 + 择时 + 调仓（等价于 09:31 my_trade）
	  ② 止损检查（等价于 14:50 check_lose，使用当日收盘价）
	  ③ 打印复盘（等价于 15:05 print_trade_info）
	"""
	g.bar_count += 1

	# 获取当前 bar 日期
	bar_date = timetag_to_datetime(C.get_bar_timetag(C.barpos), '%Y%m%d%H%M%S')

	# 前几根 bar 数据不足，跳过
	if g.bar_count < LOOKBACK_DAYS + 10:
		return

	# 首根有效 bar：一次性预加载全部行情与静态信息
	if not g.loaded:
		preload_all(C, bar_date)

	pos = g.date_pos.get(bar_date[:8])
	if pos is None:
		print(f'[回测] {bar_date} 不在预加载行情中，跳过')
		return

	print('\n' + '=' * 60)
	print(f'[回测] Bar#{g.bar_count} 日期: {bar_date}')
	print('=' * 60)

	# ============================================================
	# ① 各板块组分别选股 + 择时，然后统一调仓（等价于 09:31 my_trade）
	# ============================================================
	# 步骤1：构建股票池（布尔掩码，对应 g.stocks 各列）
	base_pool = get_stock_pool(pos)

	decisions = {}
	for group in BOARD_GROUPS:
		name = BOARD_NAMES.get(group, group)
		pool_mask = base_pool & g.board_mask[group]
		pool_size = int(pool_mask.sum())
		if pool_size == 0:
			print(f'[{name}] 股票池为空，今日不调仓')
			continue
		print(f'[{name}] 步骤1 - 股票池: {pool_size} 只')

		# 步骤2：动量打分选股，取第1名
		target_stock = get_rank(pool_mask, pos, name)
		if target_stock is None:
			print(f'[{name}] 步骤2 - 未选出目标股票')
			continue
		print(f'[{name}] 步骤2 - 目标: {target_stock} {get_name(C, target_stock)}')

		# 步骤3：计算近5日动量分数序列
		g.stock_df[group] = rank_stock_change(target_stock, pos)
		scores = g.stock_df[group].get(target_stock, [])
		print(f'[{name}] 步骤3 - 近5日动量分数: {[round(s, 4) for s in scores]}')

		# 步骤4：过滤候选股（跌停、停牌）
		target_stock = filter_target(target_stock, pos)
		if target_stock is None:
			print(f'[{name}] 步骤4 - 目标股票被过滤')
			continue
		g.today_target[group] = target_stock
		print(f'[{name}] 步骤4 - 过滤通过: {target_stock}')

		# 步骤5：计算择时信号
		signal = get_timing_signal(target_stock, group, bar_date)
		print(f'[{name}] 步骤5 - 择时信号: {signal}')
		decisions[group] = (target_stock, signal)

	# 步骤6：执行调仓（先卖后买，各组按资金占比买入）
	if decisions:
		adjust_position(decisions, C, bar_date, pos)
		print('[回测] 步骤6 - 调仓执行完毕')

	# ============================================================
	# ② 止损检查（等价于 14:50 check_lose）
	#   回测中使用当日收盘价判断
	# ============================================================
	check_lose_backtest(C, bar_date, pos)

	# ============================================================
	# ③ 打印复盘（等价于 15:05 print_trade_info）
	# ============================================================
	print_trade_info_backtest(C, bar_date, pos)


# ============================================================
# 预加载：行情宽表 + 静态信息 + RSRS
# ============================================================

def norm_date(x):
	"""行情索引统一成 YYYYMMDD 字符串"""
	return ''.join(ch for ch in str(x) if ch.isdigit())[:8]


def get_backtest_end():
	"""回测结束日期 YYYYMMDD，取不到时返回 ''（读到本地最新数据）"""
	try:
		end = str(g.ctx_end)
	except Exception:
		return ''
	digits = ''.join(ch for ch in end if ch.isdigit())
	return digits[:8] if len(digits) >= 8 else ''


def preload_all(C, bar_date):
	t0 = time.time()
	g.loaded = True

	try:
		g.ctx_end = C.end
	except Exception:
		g.ctx_end = ''
	end_time = get_backtest_end()
	start_time = (pd.Timestamp(bar_date[:8]) - pd.Timedelta(days=PRELOAD_BUFFER_DAYS)).strftime('%Y%m%d')

	# ---------- 板块成分股（只取一次） ----------
	pool_set = set()
	for sector_name in CONCEPT_SECTORS:
		try:
			for s in C.get_stock_list_in_sector(sector_name):
				pool_set.add(s)
		except Exception:
			pass
	stocks = sorted(pool_set)

	# ---------- 全市场行情一次性读取 ----------
	data = {}
	if stocks:
		try:
			data = C.get_market_data_ex(
				PRICE_FIELDS, stocks,
				period='1d',
				start_time=start_time,
				end_time=end_time,
				count=-1,
				dividend_type='none',
				fill_data=True,
				subscribe=False
			)
		except Exception as e:
			print(f'[preload] 批量获取行情失败: {e}')
			data = {}

	stocks = [s for s in stocks if s in data and data[s] is not None and len(data[s]) > 0]
	g.stocks = stocks
	g.stock_idx = {s: i for i, s in enumerate(stocks)}

	wide = {}
	if stocks:
		for f in PRICE_FIELDS:
			cols = {}
			for s in stocks:
				df = data[s]
				if f in df.columns:
					cols[s] = df[f]
			wide[f] = pd.DataFrame(cols).reindex(columns=stocks)
		dates = pd.Index([norm_date(x) for x in wide['close'].index])
		for f in PRICE_FIELDS:
			wide[f].index = [norm_date(x) for x in wide[f].index]
			wide[f] = wide[f].reindex(dates)
	else:
		dates = pd.Index([])
	del data

	g.dates = list(dates)
	g.date_pos = {d[:8]: i for i, d in enumerate(g.dates)}
	g.close = wide['close'].values.astype(float) if stocks else np.empty((0, 0))
	g.open = wide['open'].values.astype(float) if stocks else np.empty((0, 0))
	g.low = wide['low'].values.astype(float) if stocks else np.empty((0, 0))
	g.pre_close = wide['preClose'].values.astype(float) if stocks else np.empty((0, 0))
	g.suspend = wide['suspendFlag'].values.astype(float) if stocks else np.empty((0, 0))
	g.log_close = np.full(g.close.shape, np.nan)
	valid = g.close > 0
	g.log_close[valid] = np.log(g.close[valid])
	del wide

	# ---------- 静态过滤：ST + 市值（与原逻辑一致，只算一次） ----------
	static_ok = np.zeros(len(stocks), dtype=bool)
	for i, stock in enumerate(stocks):
		static_ok[i] = static_filter(C, stock)
	g.static_ok = static_ok

	# ---------- 板块归属 ----------
	boards = np.array([board_of(s) for s in stocks])
	g.board_mask = {grp: boards == grp for grp in BOARD_GROUPS}

	# ---------- RSRS 全序列 ----------
	g.rsrs = precompute_rsrs(C, end_time)

	print(f'[preload] 完成: {len(stocks)} 只股票, {len(g.dates)} 个交易日, '
		  f'静态过滤后 {int(static_ok.sum())} 只, 用时 {time.time() - t0:.1f}s')


def static_filter(C, stock):
	"""
	ST 过滤 + 市值过滤。与原版保持相同判定结果：
	  - detail 为空 → 剔除
	  - 名称含 ST → 剔除
	  - TotalValue > 0 且不在 [MIN, MAX] → 剔除
	  - 原版 TotalValue<=0 时用到未定义的 last_close，异常被吞掉后股票保留，这里同样保留
	"""
	try:
		detail = C.get_instrument_detail(stock)
		if not detail:
			return False
		g.name_cache[stock] = detail.get('InstrumentName', '')
		if 'ST' in g.name_cache[stock].upper():
			return False
		total_value = detail.get('TotalValue', 0)
		if total_value > 0 and (total_value < MIN_MARKET_CAP or total_value > MAX_MARKET_CAP):
			return False
	except Exception:
		pass
	return True


def get_name(C, stock):
	name = g.name_cache.get(stock)
	if name is None:
		try:
			name = C.get_stock_name(stock)
		except Exception:
			name = ''
		g.name_cache[stock] = name
	return name


def precompute_rsrs(C, end_time):
	"""
	一次性计算 RSRS 修正标准分，返回 {日期YYYYMMDD: 值}。
	与原版口径一致：在日期 d，使用 d 之前（不含 d）的数据，
	N 日窗口 high~low 回归斜率，最近 M 个斜率做 zscore(总体标准差)，乘以最新窗口的 R2。
	"""
	try:
		data = C.get_market_data_ex(
			['high', 'low'], [RSRS_INDEX],
			period='1d',
			end_time=end_time,
			count=-1,
			dividend_type='none',
			fill_data=True,
			subscribe=False
		)
		df = data[RSRS_INDEX]
	except Exception as e:
		print(f'[preload] RSRS 数据获取失败: {e}')
		return {}
	if df is None or len(df) == 0:
		return {}

	h = df['high'].astype(float)
	l = df['low'].astype(float)
	cov = h.rolling(RSRS_N).cov(l)
	var_l = l.rolling(RSRS_N).var()
	var_h = h.rolling(RSRS_N).var()
	beta = cov / var_l
	r2 = (cov * cov) / (var_l * var_h)
	r2 = r2.where(var_h > 0, 0.0)

	mean_b = beta.rolling(RSRS_M).mean()
	std_b = beta.rolling(RSRS_M).std(ddof=0)
	z = (beta - mean_b) / std_b
	z = z.where(std_b != 0, 0.0)
	rsrs = z * r2.where(std_b != 0, 1.0)

	# 日期 d 使用截至前一根的值；且原版要求 d 之前至少 M+N 根数据
	shifted = rsrs.shift(1)
	shifted.iloc[:RSRS_M + RSRS_N] = np.nan
	idx = [norm_date(x) for x in df.index]
	return {d: v for d, v in zip(idx, shifted.values) if not np.isnan(v)}


# ============================================================
# 取价工具（全部从内存宽表读取）
# ============================================================

def get_field(arr, stock, pos):
	i = g.stock_idx.get(stock)
	if i is None:
		return np.nan
	return arr[pos, i]


def get_close_price(C, stock, bar_date, pos):
	"""当日收盘价；股票不在预加载范围时回退到接口查询"""
	price = get_field(g.close, stock, pos)
	if not np.isnan(price):
		return float(price)
	try:
		data = C.get_market_data_ex(
			['close'], [stock], period='1d', end_time=bar_date, count=1,
			dividend_type='none', fill_data=True, subscribe=False)
		return float(data[stock]['close'].iloc[-1])
	except Exception:
		return 0.0


def get_open_price(C, stock, bar_date, pos):
	price = get_field(g.open, stock, pos)
	if not np.isnan(price):
		return float(price)
	try:
		data = C.get_market_data_ex(
			['open'], [stock], period='1d', end_time=bar_date, count=1,
			dividend_type='none', fill_data=True, subscribe=False)
		return float(data[stock]['open'].iloc[-1])
	except Exception:
		return 0.0


def board_of(stock):
	"""股票所属板块组：'main' 主板 / 'gem_star' 创业板+科创板 / 'other' 其他（如北交所）"""
	code, _, market = stock.partition('.')
	if market.upper() == 'BJ':
		return 'other'
	if code.startswith(('300', '301', '688', '689')):
		return 'gem_star'
	if code.startswith(('60', '000', '001', '002', '003')):
		return 'main'
	return 'other'


def get_limit_ratio(stock):
	"""
	根据代码前缀确定涨跌停幅度：
	  创业板(300/301)、科创板(688) → 20%
	  主板(60/00) → 10%
	"""
	code = stock.split('.')[0]
	if code.startswith(('300', '301', '688')):
		return 0.20
	return 0.10


def get_price_and_limits(stock, pos):
	"""
	获取某股票当日开盘价、涨停价、跌停价、最低价。
	拿不到时返回 (0.0, 0.0, 0.0, 0.0)。
	"""
	open_price = get_field(g.open, stock, pos)
	if np.isnan(open_price):
		return 0.0, 0.0, 0.0, 0.0
	open_price = float(open_price)
	pre_close = get_field(g.pre_close, stock, pos)
	pre_close = open_price if np.isnan(pre_close) else float(pre_close)
	low_price = float(get_field(g.low, stock, pos))
	if pre_close <= 0:
		return open_price, 0.0, 0.0, low_price

	ratio = get_limit_ratio(stock)
	limit_up = round(pre_close * (1 + ratio), 2)
	limit_down = round(pre_close * (1 - ratio), 2)
	return open_price, limit_up, limit_down, low_price


# ============================================================
# 回测版止损检查
# ============================================================

def check_lose_backtest(C, bar_date, pos):
	"""
	回测版止损：使用当日收盘价判断是否触发 -15% 硬止损
	"""
	holdings = get_trade_detail_data(g.account, g.acct_type, 'position')
	if not holdings:
		return

	for p in holdings:
		stock = p.m_strInstrumentID + '.' + p.m_strExchangeID
		cost_price = p.m_dOpenPrice
		volume = p.m_nCanUseVolume
		if volume <= 0 or cost_price <= 0:
			continue

		current_price = get_close_price(C, stock, bar_date, pos)
		if current_price <= 0:
			continue

		profit_ratio = (current_price - cost_price) / cost_price
		print(f'[止损检查] {stock} {get_name(C, stock)} '
			  f'成本:{cost_price:.2f} 收盘:{current_price:.2f} 盈亏:{profit_ratio:.2%}')

		if profit_ratio <= STOP_LOSS_RATIO:
			print(f'[止损检查] {stock} 触发硬止损！盈亏 {profit_ratio:.2%} <= -15%，强制清仓')
			msg = f'硬止损平仓 {stock}'
			passorder(24, 1101, g.account, stock, 5, -1, volume,
					  STRATEGY_NAME, 1, msg, C)


# ============================================================
# 回测版复盘打印
# ============================================================

def print_trade_info_backtest(C, bar_date, pos):
	"""
	回测版复盘：打印当日成交、持仓、资金
	"""
	deals = get_trade_detail_data(g.account, g.acct_type, 'deal')
	if deals:
		today_deals = []
		for deal in deals:
			deal_date = deal.m_strTradeDate.replace('-', '')
			if deal_date == bar_date[:8]:
				today_deals.append(deal)
		if today_deals:
			print(f'--- 今日成交 ({len(today_deals)} 笔) ---')
			for deal in today_deals:
				print(f'  {deal.m_strInstrumentID}.{deal.m_strExchangeID} '
					  f'{"买入" if deal.m_nDirection == 1 else "卖出"} '
					  f'价格:{deal.m_dPrice:.2f} 数量:{deal.m_nVolume}')

	holdings = get_trade_detail_data(g.account, g.acct_type, 'position')
	if holdings:
		for p in holdings:
			stock = p.m_strInstrumentID + '.' + p.m_strExchangeID
			cost_price = p.m_dOpenPrice
			volume = p.m_nCanUseVolume
			if volume <= 0:
				continue

			current_price = get_close_price(C, stock, bar_date, pos)
			market_value = current_price * volume
			profit_ratio = (current_price - cost_price) / cost_price if cost_price > 0 else 0

			print(f'  持仓: {stock} {get_name(C, stock)} '
				  f'成本:{cost_price:.2f} 收盘:{current_price:.2f} '
				  f'盈亏:{profit_ratio:.2%} 市值:{market_value:.0f}')

	acc = get_trade_detail_data(g.account, g.acct_type, 'account')
	if acc:
		acc = acc[0]
		print(f'  资金: 可用={acc.m_dAvailable:.0f} 总资产={acc.m_dBalance:.0f}')


# ============================================================
# 步骤1：构建股票池 get_stock_pool()
# ============================================================

def get_stock_pool(pos):
	"""
	返回布尔掩码（对应 g.stocks）：静态过滤通过 + 当日有行情 + 当日未停牌
	"""
	has_data = ~np.isnan(g.close[pos])
	not_suspend = g.suspend[pos] != 1
	return g.static_ok & has_data & not_suspend


# ============================================================
# 步骤2：动量打分选股 get_rank()
# ============================================================

def momentum_scores(log_window):
	"""
	向量化动量分数：log_window 形状 (L, K)，每列一只股票的对数收盘价。
	分数 = (exp(slope * 244) - 1) * R2，R2<=0 时为 0；与逐只 np.polyfit 结果一致。
	"""
	n = log_window.shape[0]
	x = np.arange(n, dtype=float)
	xc = x - x.mean()
	sxx = np.dot(xc, xc)
	yc = log_window - log_window.mean(axis=0)
	sxy = xc @ yc
	sst = np.einsum('ij,ij->j', yc, yc)
	slope = sxy / sxx
	with np.errstate(divide='ignore', invalid='ignore'):
		r2 = np.where(sst > 0, slope * sxy / sst, 0.0)
	score = (np.exp(slope * TRADING_DAYS_PER_YEAR) - 1) * np.abs(r2)
	return np.where(r2 <= 0, 0.0, score)


def get_rank(pool_mask, pos, name=''):
	"""
	使用 bar 之前的 LOOKBACK_DAYS 根收盘价打分（不含当前 bar），选第1名
	"""
	if pos < LOOKBACK_DAYS:
		return None
	window = g.log_close[pos - LOOKBACK_DAYS:pos]
	mask = pool_mask & ~np.isnan(window).any(axis=0)
	cols = np.nonzero(mask)[0]
	if len(cols) == 0:
		return None

	scores = momentum_scores(window[:, cols])
	ok = ~np.isnan(scores)
	cols, scores = cols[ok], scores[ok]
	if len(cols) == 0:
		return None

	order = np.argsort(-scores, kind='stable')
	top = [(g.stocks[cols[k]], round(float(scores[k]), 4)) for k in order[:3]]
	if VERBOSE:
		print(f'[{name}] sorted_stocks', [(g.stocks[cols[k]], float(scores[k])) for k in order])
	print(f'[get_rank][{name}] Top3: {top}')
	return g.stocks[cols[order[0]]]


# ============================================================
# 步骤3：计算近5日动量分数序列 rank_stock_change()
# ============================================================

def rank_stock_change(stock, pos):
	"""
	目标股票近6个时点的动量分数（从远到近），所有窗口均不含当前 bar
	"""
	i = g.stock_idx.get(stock)
	if i is None:
		return {}
	col = g.log_close[:pos + 1, i]
	# 原版要求至少 LOOKBACK_DAYS + 2 根数据
	if np.count_nonzero(~np.isnan(g.close[:pos + 1, i])) < LOOKBACK_DAYS + 2:
		return {}

	scores = []
	for k in range(5, -1, -1):
		end = pos - k
		start = end - LOOKBACK_DAYS
		if start < 0:
			scores.append(0.0)
			continue
		w = col[start:end]
		if np.any(np.isnan(w)):
			scores.append(0.0)
		else:
			scores.append(float(momentum_scores(w.reshape(-1, 1))[0]))
	return {stock: scores}


# ============================================================
# 步骤4：过滤候选股
# ============================================================

def filter_target(stock, pos):
	"""
	剔除跌停、停牌（使用当日数据）
	"""
	if stock is None:
		return None

	if get_field(g.suspend, stock, pos) == 1:
		print(f'[filter_target] {stock} 停牌中')
		return None

	last_close = get_field(g.close, stock, pos)
	if np.isnan(last_close) or last_close <= 0:
		return None
	pre_close = get_field(g.pre_close, stock, pos)
	if np.isnan(pre_close):
		pre_close = last_close

	limit_down = round(pre_close * (1 - get_limit_ratio(stock)), 2)
	if last_close <= limit_down:
		print(f'[filter_target] {stock} 跌停 收盘:{last_close} 跌停价:{limit_down}')
		return None

	return stock


# ============================================================
# 步骤5：计算综合择时信号 get_timing_signal()
# ============================================================

def get_timing_signal(stock, group, bar_date):
	"""
	择时信号：
	  RSRS 仅记录，不介入决策
	  实际信号 = 动量分数连续下降天数是否 >= 2
	"""
	rsrs = g.rsrs.get(bar_date[:8])
	if rsrs is not None:
		print(f'[择时] RSRS修正标准分: {rsrs:.4f}')
	else:
		print('[择时] RSRS 数据不足')

	scores = g.stock_df.get(group, {}).get(stock, [])
	if len(scores) < 1:
		return 'KEEP'

	sig = 0
	for i in range(len(scores) - 1, 0, -1):
		if scores[i] < scores[i - 1]:
			sig += 1
		else:
			break

	print(f'[择时] 动量分数序列: {[round(s, 4) for s in scores]}')
	print(f'[择时] 连续下降天数: {sig}')

	return 'SELL' if sig >= 2 else 'BUY'


# ============================================================
# 步骤6：执行调仓 adjust_position()
# ============================================================

def adjust_position(decisions, C, bar_date, pos):
	"""
	调仓，decisions = {板块组: (目标股, 信号)}，每组只管理本板块的持仓：
	  SELL：清掉本组持仓
	  BUY/KEEP：本组持仓不是目标股 → 换仓；是目标股 → 持有
	先执行所有卖出，再按 BOARD_GROUPS 资金占比买入。
	组预算 = 总资产(按今日开盘价) * 占比，且不超过当前可用资金。
	"""
	holdings = get_trade_detail_data(g.account, g.acct_type, 'position')
	current_holdings = {}
	for p in holdings:
		s = p.m_strInstrumentID + '.' + p.m_strExchangeID
		vol = p.m_nCanUseVolume
		if vol > 0:
			current_holdings[s] = vol
	print(f'[调仓] 当前持仓: {current_holdings}')

	acc_info = get_trade_detail_data(g.account, g.acct_type, 'account')
	if not acc_info:
		print('[调仓] 无法获取账户信息')
		return
	total_asset = float(acc_info[0].m_dAvailable)
	for s, vol in current_holdings.items():
		total_asset += get_open_price(C, s, bar_date, pos) * vol

	# ---------- 卖出 ----------
	buys = []
	for group, (stock, signal) in decisions.items():
		name = BOARD_NAMES.get(group, group)
		group_holdings = {s: v for s, v in current_holdings.items() if board_of(s) == group}

		if signal == 'SELL':
			for s, vol in group_holdings.items():
				msg = f'SELL信号 清仓 {s}'
				print(f'[调仓][{name}] {msg}')
				passorder(24, 1101, g.account, s, 11, get_open_price(C, s, bar_date, pos), vol,
						  STRATEGY_NAME, 1, msg, C)
			continue

		# BUY / KEEP：已是目标股则持有
		if group_holdings.get(stock, 0) > 0:
			print(f'[调仓][{name}] KEEP: 继续持有 {stock}')
			continue

		# 换仓：先卖旧
		for s, vol in group_holdings.items():
			msg = f'切换标的 卖出 {s}'
			print(f'[调仓][{name}] {msg}')
			passorder(24, 1101, g.account, s, 11, get_open_price(C, s, bar_date, pos), vol,
					  STRATEGY_NAME, 1, msg, C)
		buys.append((group, stock))

	if not buys:
		return

	# ---------- 买入 ----------
	acc_info = get_trade_detail_data(g.account, g.acct_type, 'account')
	if not acc_info:
		print('[调仓] 无法获取账户信息')
		return
	available_cash = int(acc_info[0].m_dAvailable)

	for group, stock in buys:
		name = BOARD_NAMES.get(group, group)
		budget = min(total_asset * BOARD_GROUPS[group], available_cash)

		current_price, limit_up, limit_down, low_price = get_price_and_limits(stock, pos)
		if current_price <= 0:
			print(f'[调仓][{name}] {stock} 价格异常: {current_price}')
			continue

		buy_vol = int(budget / current_price / 100) * 100
		if buy_vol < 100:
			print(f'[调仓][{name}] 资金不足买1手，预算:{budget:.0f} 股价:{current_price}')
			continue

		if low_price >= limit_up:
			print(f'[调仓][{name}] {stock} 开盘涨停，无法买入')
			continue

		msg = f'BUY信号 买入 {stock} {buy_vol}股'
		print(f'[调仓][{name}] {msg}')
		passorder(23, 1101, g.account, stock, 11, current_price, buy_vol,
				  STRATEGY_NAME, 1, msg, C)
		available_cash -= buy_vol * current_price


# ============================================================
# stop 函数
# ============================================================

def stop(C):
	print(f'[动量择时策略-回测版] 回测结束，共处理 {g.bar_count} 根K线')
