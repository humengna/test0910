# coding: gbk
"""
QMT 版本：配对交易（伊利股份 600887 / 招商银行 600036）
移植自聚宽策略：https://www.joinquant.com/post/11593
（原文：https://www.joinquant.com/post/1810 「配对交易：用协整做搬砖」）

策略逻辑（与原聚宽代码保持一致）：
  1. 取两只股票过去 test_days(120) 个交易日收盘价（不含当天）
  2. 价差序列 spread = P2 - ratio * P1，计算最新一天价差的 z-score
  3. z >  1          -> 'buy1'  : 清仓股票2，全仓股票1
     z < -1          -> 'buy2'  : 清仓股票1，全仓股票2
     0 <= z <= 1     -> 'side1'
     -1 <= z < 0     -> 'side2'
  4. 当前 buy1 且新信号 side2，或当前 buy2 且新信号 side1（z 穿越 0 轴）
     -> 两只股票按 p / q 比例（默认各 50%）持有，状态变为 'even'

使用方法：
  - QMT 新建 Python 策略，把本文件内容粘贴进去（QMT 编辑器按 GBK 保存，
    所以第一行必须是 # coding: gbk，不能写 utf-8）
  - 回测参数：周期选「日线」，主图品种随意（如 600036.SH），
    设置回测起止时间、初始资金、手续费、滑点（原策略无滑点）
  - 回测前先在「数据管理」中补充两只股票的日线数据（含除权数据）
"""
import numpy as np


class _G(object):
    pass


g = _G()


# ============================================================
def init(ContextInfo):
    set_params()
    set_variables()
    set_backtest(ContextInfo)


# ---代码块1. 设置参数
def set_params():
    g.security1 = '600887.SH'      # 股票1
    g.security2 = '600036.SH'      # 股票2
    g.benchmark = '600036.SH'      # 基准
    g.regression_ratio = 1.000     # 回归系数
    g.p = 0.5                      # 股票1默认仓位
    g.q = 0.5                      # 股票2默认仓位
    g.test_days = 120              # 计算 z-score 的天数
    # 成交价：True 用当日开盘价下单（与聚宽日线回测 handle_data 开盘成交一致）
    #        False 用最新价（回测中即当日收盘价）
    g.use_open_price = True
    # 下单时预留的手续费比例，防止资金不足导致废单
    g.fee_buffer = 0.0005


# ---代码块2. 设置变量
def set_variables():
    g.state = 'empty'
    g.last_date = None


# ---代码块3. 设置回测
def set_backtest(ContextInfo):
    # 回测时 accID 为空则使用 'test'；实盘请在策略界面选择账号
    g.acct = getattr(ContextInfo, 'accID', '') or 'test'
    ContextInfo.set_account(g.acct)
    ContextInfo.set_universe([g.security1, g.security2])
    ContextInfo.benchmark = g.benchmark


# ============================================================
def handlebar(ContextInfo):
    # 实盘/模拟时只在最新 K 线上运行，避免在历史 K 线上重复下单
    if not ContextInfo.do_back_test and not ContextInfo.is_last_bar():
        return

    today = timetag_to_datetime(ContextInfo.get_bar_timetag(ContextInfo.barpos), '%Y%m%d')
    # 每个交易日只运行一次（与聚宽日线 handle_data 一致）
    if today == g.last_date:
        return
    g.last_date = today

    new_state = get_signal(ContextInfo, today)
    if new_state is None:
        return
    change_positions(new_state, ContextInfo, today)


# ---代码块4. 计算 z-score
def get_close_history(ContextInfo, code, today, count):
    """
    取 today 之前 count 个交易日收盘价（不含 today），模拟聚宽
    use_real_price 下 attribute_history 的「动态前复权」：
    等比前复权序列，再整体缩放使最后一天等于真实收盘价。
    """
    def _fetch(dividend_type):
        data = ContextInfo.get_market_data_ex(
            ['close'], [code], period='1d', end_time=today,
            count=count + 1, dividend_type=dividend_type,
            fill_data=True, subscribe=False)
        df = data.get(code)
        if df is None or len(df) == 0:
            return None
        df = df[[str(i)[:8] < today for i in df.index]]
        return df['close'].tail(count)

    adj = _fetch('front_ratio')
    raw = _fetch('none')
    if adj is None or raw is None or len(adj) < count or len(raw) < count:
        return None
    adj = np.array(adj, dtype=float)
    raw_last = float(raw.iloc[-1])
    if adj[-1] <= 0 or raw_last <= 0:
        return None
    return adj * (raw_last / adj[-1])


def z_test(ContextInfo, today):
    prices1 = get_close_history(ContextInfo, g.security1, today, g.test_days)
    prices2 = get_close_history(ContextInfo, g.security2, today, g.test_days)
    if prices1 is None or prices2 is None:
        return None
    # 根据回归比例算出平稳序列 Y - a.X
    stable_series = prices2 - g.regression_ratio * prices1
    series_mean = np.mean(stable_series)
    sigma = np.std(stable_series)
    if sigma == 0:
        return None
    diff = stable_series[-1] - series_mean
    return diff / sigma


# ---代码块5. 获取信号
def get_signal(ContextInfo, today):
    z_score = z_test(ContextInfo, today)
    if z_score is None:
        return None
    if z_score > 1:
        return 'buy1'
    if z_score < -1:
        return 'buy2'
    if z_score >= 0:
        return 'side1'
    return 'side2'


# ---代码块6. 根据信号调整仓位
def change_positions(new_state, ContextInfo, today):
    if new_state == 'buy1':
        # 全卖股票2，全买股票1
        rebalance(ContextInfo, today, {g.security1: 1.0, g.security2: 0.0})
        g.state = 'buy1'
    elif new_state == 'buy2':
        # 全卖股票1，全买股票2
        rebalance(ContextInfo, today, {g.security1: 0.0, g.security2: 1.0})
        g.state = 'buy2'
    elif (g.state == 'buy1' and new_state == 'side2') or \
         (g.state == 'buy2' and new_state == 'side1'):
        # z-score 穿越 0 轴，按 p、q 恢复默认仓位
        rebalance(ContextInfo, today, {g.security1: g.p, g.security2: g.q})
        g.state = 'even'


# ============================================================
# 交易工具函数
def get_prices(ContextInfo, today):
    field = 'open' if g.use_open_price else 'close'
    codes = [g.security1, g.security2]
    # 回测撮合使用主图的复权方式，下单价必须同口径，否则会超出当根 K 线
    # 最高最低价而被改成最新价成交，所以这里用 'follow'（跟随主图）
    data = ContextInfo.get_market_data_ex(
        [field, 'high', 'low'], codes, period='1d', end_time=today, count=1,
        dividend_type='follow', fill_data=True, subscribe=False)
    prices = {}
    for code in codes:
        df = data.get(code)
        if df is None or len(df) == 0 or str(df.index[-1])[:8] != today:
            return None  # 当天停牌/无数据
        px = float(df[field].iloc[-1])
        if not px > 0:
            return None
        # 保险：限制在当根 K 线最高最低价之间
        px = min(max(px, float(df['low'].iloc[-1])), float(df['high'].iloc[-1]))
        prices[code] = px
    return prices


def get_account_info():
    cash, positions = 0.0, {}
    accounts = get_trade_detail_data(g.acct, 'stock', 'account')
    if accounts:
        cash = accounts[0].m_dAvailable
    for pos in get_trade_detail_data(g.acct, 'stock', 'position'):
        code = pos.m_strInstrumentID + '.' + pos.m_strExchangeID
        positions[code] = (pos.m_nVolume, pos.m_nCanUseVolume)
    return cash, positions


def rebalance(ContextInfo, today, weights):
    """按目标权重调仓：先卖后买，买入量受可用资金限制，A 股 100 股整数倍"""
    prices = get_prices(ContextInfo, today)
    if prices is None:
        print('%s 停牌或无行情，跳过调仓' % today)
        return
    cash, positions = get_account_info()
    total_value = cash + sum(positions.get(c, (0, 0))[0] * prices[c] for c in prices)

    # 先卖
    for code, w in weights.items():
        vol, can_use = positions.get(code, (0, 0))
        target = int(total_value * w / prices[code] / 100) * 100
        if vol > target:
            # 目标为 0 时全部卖出（含零股），否则按整手卖
            sell = vol - target if target == 0 else int((vol - target) / 100) * 100
            sell = min(sell, can_use)
            if sell > 0:
                order(ContextInfo, 24, code, prices[code], sell)
                cash += sell * prices[code] * (1 - g.fee_buffer)

    # 再买
    for code, w in weights.items():
        vol = positions.get(code, (0, 0))[0]
        target = int(total_value * w / prices[code] / 100) * 100
        if target > vol:
            affordable = int(cash / (prices[code] * (1 + g.fee_buffer)) / 100) * 100
            buy = min(int((target - vol) / 100) * 100, affordable)
            if buy > 0:
                order(ContextInfo, 23, code, prices[code], buy)
                cash -= buy * prices[code] * (1 + g.fee_buffer)


def order(ContextInfo, op_type, code, price, volume):
    # op_type: 23 买入 / 24 卖出；1101 单股单账号按股数；
    # prType 11 指定价，5 最新价
    pr_type = 11 if g.use_open_price else 5
    passorder(op_type, 1101, g.acct, code, pr_type, price, volume,
              'pair_trading', 1, '', ContextInfo)
    print('%s %s %s %d 股 @ %.2f' % (
        g.last_date, '买入' if op_type == 23 else '卖出', code, volume, price))
