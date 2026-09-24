# coding: gbk
"""
配对信号策略 —— QMT 版（动态选股，不固定配对）

与 qmt_pair_trading.py 的区别：
  - 不再固定两只股票。策略定期（默认每 20 个交易日）用过去约 2 年的数据，
    在整个股票池里重新做协整选股，得到一个「候选配对池」
  - 每个交易日检查候选池里所有配对的 z 值：某一边被显著低估（|z| > 入场阈值）
    就产生买入信号，买入被低估的那只股票
  - 同一只股票可能同时被多个配对判定为低估：信号越多、偏离越大，排名越靠前
  - 每只股票单独持仓、单独退出：价差回归、止损、超过最长持有天数、
    或配对关系失效（重新检验不再协整）时卖出
  - 只用当天之前的数据，回测天然是「滚动样本外」，没有未来数据

价差用对数价格：spread = ln(P2) - beta * ln(P1)。两只股票同比例涨跌时价差不变，
避免用价格直接相减时「整体涨跌被误判成相对变化」的问题；也不再需要动态前复权。

使用方法：
  1. 新建 Python 策略，粘贴本文件全部内容（第一行保持 # coding: gbk）
  2. 修改 set_params() 里的股票池和参数
  3. 数据管理里下载股票池的日线和除权数据，起始时间比回测开始早约 3 年
  4. 回测：周期选「日线」，设置回测区间、资金、费率。实盘/模拟：同样日线运行
  5. g.trade = False 时只输出信号不下单（每日信号保存在 g.signal_file）
"""
import math
import os
import itertools
import time

import numpy as np
import pandas as pd


class _G(object):
    pass


g = _G()


def set_params():
    # ---- 股票池：sector 或 codes 二选一（codes 非空时优先）----
    g.sector = '沪深300'
    g.codes = []
    # ---- 选股（定期重新做协整检验，得到候选配对池）----
    g.formation_days = 500          # 选股用的历史长度（交易日，约 2 年）
    g.reselect_every = 20           # 每隔多少个交易日重新选股（1 = 每天，很慢）
    g.min_corr = 0.85               # 对数价格相关系数下限（初筛）
    g.max_candidates = 3000         # 初筛后最多检验多少对（按相关系数取前 N，控制速度）
    g.max_pvalue = 0.05             # 协整 p 值上限
    g.min_half_life = 2             # 半衰期范围（交易日）
    g.max_half_life = 40
    g.pool_size = 60                # 候选配对池最多保留多少对（按 p 值）
    g.min_amount = 5e7              # 选股期日均成交额下限（元）
    g.max_missing = 0.05            # 选股期允许的缺失/停牌比例
    # ---- 信号 ----
    g.window = 120                  # 计算 z 值的窗口（交易日）
    g.entry_z = 2.0                 # |z| 超过它产生买入信号
    g.exit_z = 0.0                  # z 回到它（穿越均值）就卖出
    # ---- 持仓与风控 ----
    g.max_positions = 5             # 最多同时持有几只，每只约 1/N 资金
    g.stop_loss = 0.15              # 从买入价下跌超过 15% 止损
    g.max_hold_days = 60            # 最长持有天数（交易日）
    g.break_pvalue = 0.2            # 重新选股时，持仓配对 p 值高于它视为关系失效，卖出
    g.cooldown_days = 20            # 止损/超时卖出后，这只股票多少天内不再买入
    # 市场过滤：指数收盘价低于 N 日均线时不开新仓（None 表示不启用）
    g.market_index = None           # 如 '000300.SH'
    g.market_ma = 60
    # ---- 执行 ----
    g.trade = True                  # False：只输出信号，不下单
    g.fee_buffer = 0.0005
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    base = desktop if os.path.isdir(desktop) else os.path.expanduser('~')
    g.signal_file = os.path.join(base, 'pair_signals.csv')


# ============================================================
def init(ContextInfo):
    set_params()
    g.acct = getattr(ContextInfo, 'accID', '') or 'test'
    try:
        ContextInfo.set_account(g.acct)
    except Exception:
        pass
    g.data = None          # 行情缓存
    g.pool = []            # 候选配对池：dict(x, y, ix, iy, beta, p, hl)
    g.last_select_date = None
    g.meta = {}            # 持仓信息：code -> dict(pair, side, entry_px, entry_date)
    g.cooldown = {}        # code -> 该日期（含）之前不再买入
    g.last_date = None
    g.signal_rows = []


def handlebar(ContextInfo):
    if not ContextInfo.do_back_test and not ContextInfo.is_last_bar():
        return
    today = timetag_to_datetime(ContextInfo.get_bar_timetag(ContextInfo.barpos), '%Y%m%d')
    if today == g.last_date:
        return
    g.last_date = today
    try:
        run_day(ContextInfo, today)
    except Exception as e:
        import traceback
        print('%s 运行出错: %s' % (today, e))
        print(traceback.format_exc())


# ============================================================
# 协整检验（numpy 实现，与 statsmodels.tsa.stattools.coint 结果一致）
# ============================================================
def _ols(y, X):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X.dot(coef)


def _lagmat(x, maxlag):
    n = len(x)
    return np.column_stack([x[maxlag - j:n - j] for j in range(maxlag + 1)])


def adf_tstat(u):
    u = np.asarray(u, dtype=float)
    nobs = len(u)
    maxlag = int(math.ceil(12.0 * (nobs / 100.0) ** 0.25))
    maxlag = max(0, min(nobs // 2 - 1, maxlag))
    du = np.diff(u)

    def design(lag):
        dl = _lagmat(du, lag)
        n = dl.shape[0]
        return dl[:, 0], np.column_stack([u[-n - 1:-1]] + [dl[:, j] for j in range(1, lag + 1)])

    y_full, X_full = design(maxlag)
    n = len(y_full)
    L = np.linalg.cholesky(X_full.T.dot(X_full))
    qy = np.linalg.solve(L, X_full.T.dot(y_full))
    ssr = np.maximum(y_full.dot(y_full) - np.cumsum(qy ** 2), 1e-300)
    k = np.arange(1, maxlag + 2)
    aic = n * (math.log(2 * math.pi) + np.log(ssr / n) + 1) + 2 * k
    y, X = design(int(np.argmin(aic)))
    coef, r = _ols(y, X)
    sigma2 = r.dot(r) / (len(y) - X.shape[1])
    return coef[0] / math.sqrt(sigma2 * np.linalg.inv(X.T.dot(X))[0, 0])


def mackinnon_p(tstat):
    if tstat > 0.92:
        return 1.0
    if tstat < -18.86:
        return 0.0
    c = [2.92, 1.5012, 0.039796] if tstat <= -2.62 else [2.1945, 0.64695, -0.29198, -0.042377]
    v = sum(ci * tstat ** i for i, ci in enumerate(c))
    return 0.5 * (1 + math.erf(v / math.sqrt(2)))


def coint_test(y, x):
    """返回 (p 值, beta, 残差)"""
    coef, resid = _ols(y, np.column_stack([np.ones_like(x), x]))
    return mackinnon_p(adf_tstat(resid)), coef[1], resid


def half_life(resid):
    coef, _ = _ols(np.diff(resid), np.column_stack([np.ones(len(resid) - 1), resid[:-1]]))
    return -math.log(2) / coef[1] if coef[1] < 0 else np.inf


# ============================================================
# 数据
# ============================================================
def load_data(ContextInfo, end):
    """一次性读取股票池全部历史，之后按日期序号切片（只用当天之前的行）"""
    codes = list(g.codes) or ContextInfo.get_stock_list_in_sector(g.sector)
    if not codes:
        print('股票池为空：请检查板块名称 %s 或填写 g.codes' % g.sector)
        return None
    t0 = time.time()
    start = '20000101'

    def wide(fields, dividend_type, cs):
        res = ContextInfo.get_market_data_ex(fields, cs, period='1d', start_time=start,
                                             end_time=end, count=-1,
                                             dividend_type=dividend_type,
                                             fill_data=False, subscribe=False)
        out = {}
        for f in fields:
            cols = {}
            for c in cs:
                d = res.get(c)
                if d is not None and len(d) > 0:
                    s = d[f].copy()
                    s.index = [str(i)[:8] for i in s.index]
                    cols[c] = s
            out[f] = pd.DataFrame(cols).sort_index()
        return out

    adj = wide(['close'], 'front_ratio', codes)
    close = adj['close']
    if close.empty:
        print('没有读到行情，请先在「数据管理」下载股票池日线和除权数据')
        return None
    codes = list(close.columns)
    dates = list(close.index)
    follow = wide(['open'], 'follow', codes)['open'].reindex(index=dates, columns=codes)
    raw = wide(['volume', 'amount'], 'none', codes)
    vol = raw['volume'].reindex(index=dates, columns=codes)
    amt = raw['amount'].reindex(index=dates, columns=codes)
    st = np.zeros(len(codes), dtype=bool)
    for i, c in enumerate(codes):
        try:
            st[i] = 'ST' in str(ContextInfo.get_instrumentdetail(c).get('InstrumentName', '')).upper()
        except Exception:
            pass
    mkt = None
    if g.market_index:
        m = wide(['close'], 'none', [g.market_index])['close']
        if g.market_index in m:
            mkt = m[g.market_index].reindex(dates).ffill().values.astype(float)
    valid = close.notna().values & (vol.fillna(0).values > 0)
    logc = np.log(close.ffill().values.astype(float))
    print('读取 %d 只股票、%d 个交易日行情，用时 %.1f 秒' % (len(codes), len(dates), time.time() - t0))
    return {
        'codes': codes, 'col': dict((c, i) for i, c in enumerate(codes)),
        'dates': dates, 'idx': dict((d, i) for i, d in enumerate(dates)),
        'logc': logc,                                    # 对数前复权收盘价（已向前填充）
        'valid': valid,                                  # 当天有真实交易
        'open': follow.values.astype(float),             # 下单用开盘价（跟随主图复权）
        'amount': amt.values.astype(float),
        'st': st, 'mkt': mkt,
    }


def ensure_data(ContextInfo, today):
    """回测：读一次到最新；实盘：当天不在缓存里就重新读"""
    if g.data is None or today not in g.data['idx']:
        end = pd.Timestamp.today().strftime('%Y%m%d') if ContextInfo.do_back_test else today
        g.data = load_data(ContextInfo, end)
        if g.data is not None:
            remap()
    return g.data is not None and today in g.data['idx']


def remap():
    """重新读数据后股票列顺序可能变化，按代码重新定位配对里的列号"""
    col = g.data['col']
    for pair in g.pool + [m['pair'] for m in g.meta.values()]:
        pair['ix'], pair['iy'] = col.get(pair['x'], -1), col.get(pair['y'], -1)
    g.pool = [pr for pr in g.pool if pr['ix'] >= 0 and pr['iy'] >= 0]


def days_between(d1, t):
    """日期 d1 到第 t 个交易日之间的交易日数"""
    i = g.data['idx'].get(d1)
    return t - i if i is not None else 10 ** 6


# ============================================================
# 选股：在 [t-formation, t) 上做协整检验，得到候选配对池
# ============================================================
def select_pairs(t):
    d = g.data
    lo = t - g.formation_days
    if lo < 0:
        return []
    L = d['logc'][lo:t]
    valid = d['valid'][lo:t]
    ok = (valid.mean(axis=0) >= 1 - g.max_missing) & ~d['st'] & np.isfinite(L).all(axis=0)
    amt = np.nanmean(d['amount'][lo:t], axis=0)
    ok &= np.nan_to_num(amt) >= g.min_amount
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return []
    corr = np.corrcoef(L[:, idx].T)
    iu = np.triu_indices(len(idx), 1)
    cv = corr[iu]
    sel = np.where(cv >= g.min_corr)[0]
    sel = sel[np.argsort(-cv[sel])][:g.max_candidates]
    pool = []
    for k in sel:
        a, b = idx[iu[0][k]], idx[iu[1][k]]
        best = None
        for ix, iy in ((a, b), (b, a)):
            try:
                p, beta, resid = coint_test(L[:, iy], L[:, ix])
            except Exception:
                continue
            if best is None or p < best[0]:
                best = (p, beta, resid, ix, iy)
        if best is None:
            continue
        p, beta, resid, ix, iy = best
        if p > g.max_pvalue or beta <= 0:
            continue
        hl = half_life(resid)
        if not (g.min_half_life <= hl <= g.max_half_life):
            continue
        pool.append({'ix': ix, 'iy': iy, 'x': d['codes'][ix], 'y': d['codes'][iy],
                     'beta': beta, 'p': p, 'hl': hl})
    pool.sort(key=lambda r: r['p'])
    return pool[:g.pool_size]


def pair_pvalue(pair, t):
    L = g.data['logc'][t - g.formation_days:t]
    try:
        return coint_test(L[:, pair['iy']], L[:, pair['ix']])[0]
    except Exception:
        return 1.0


def zscore(pair, t):
    """用 [t-window, t) 的对数价差计算 z（不含当天）"""
    L = g.data['logc'][t - g.window:t]
    s = L[:, pair['iy']] - pair['beta'] * L[:, pair['ix']]
    if not np.isfinite(s).all():
        return None
    sd = s.std()
    return None if sd <= 0 else (s[-1] - s.mean()) / sd


# ============================================================
# 每日流程
# ============================================================
def run_day(ContextInfo, today):
    if not ensure_data(ContextInfo, today):
        return
    d = g.data
    t = d['idx'][today]
    if t < max(g.formation_days, g.window):
        return

    # 1. 定期重新选股；同时检查持仓配对的关系是否失效
    broken = set()
    if g.last_select_date is None or days_between(g.last_select_date, t) >= g.reselect_every:
        t0 = time.time()
        g.pool = select_pairs(t)
        g.last_select_date = today
        for code, m in g.meta.items():
            if m['pair']['ix'] < 0 or pair_pvalue(m['pair'], t) > g.break_pvalue:
                broken.add(code)
        print('%s 重新选股：候选配对 %d 对，用时 %.1f 秒' % (today, len(g.pool), time.time() - t0))

    # 2. 当前持仓（只管理本策略买入的股票；只出信号时用虚拟持仓）
    cash, holding = get_account() if g.trade else (0.0, {})
    if g.trade:
        for code in list(g.meta.keys()):
            if holding.get(code, (0, 0))[0] <= 0 and g.meta[code]['entry_date'] != today:
                del g.meta[code]    # 买单没成交

    px = d['open'][t]

    # 3. 卖出信号
    sells = []
    for code, m in g.meta.items():
        i = d['col'].get(code)
        if i is None or m['pair']['ix'] < 0:
            if code in broken:
                sells.append((code, '已不在股票池'))
            continue
        z = zscore(m['pair'], t)
        ret = px[i] / m['entry_px'] - 1 if px[i] > 0 else 0.0
        reason = None
        if code in broken:
            reason = '配对关系失效'
        elif ret <= -g.stop_loss:
            reason = '止损 %.1f%%' % (ret * 100)
        elif days_between(m['entry_date'], t) >= g.max_hold_days:
            reason = '持有超过 %d 天' % g.max_hold_days
        elif z is not None and ((m['side'] == 'y' and z >= -g.exit_z) or
                                (m['side'] == 'x' and z <= g.exit_z)):
            reason = '价差回归 z=%.2f' % z
        if reason:
            sells.append((code, reason))
            if not reason.startswith('价差回归'):
                g.cooldown[code] = t + g.cooldown_days - 1

    # 4. 买入信号：被低估的一方。同一只股票的多个信号合并打分
    market_ok = True
    if d['mkt'] is not None and t > g.market_ma:
        m = d['mkt'][t - g.market_ma:t]
        market_ok = bool(np.isfinite(m).all() and m[-1] >= m.mean())
    cand = {}
    for pair in g.pool:
        z = zscore(pair, t)
        if z is None or abs(z) < g.entry_z:
            continue
        # z < 0：y 相对便宜，买 y；z > 0：x 相对便宜，买 x
        side = 'y' if z < 0 else 'x'
        code = pair[side]
        c = cand.setdefault(code, {'n': 0, 'score': 0.0, 'best': None})
        c['n'] += 1
        c['score'] += abs(z)
        if c['best'] is None or abs(z) > abs(c['best'][2]):
            c['best'] = (pair, side, z)
    sold = set(c for c, _ in sells)
    ranked = sorted(cand.items(), key=lambda kv: (-kv[1]['n'], -kv[1]['score']))
    buys = []
    slots = g.max_positions - (len(g.meta) - len(sold))
    for code, c in ranked:
        if slots <= 0 or not market_ok:
            break
        i = d['col'][code]
        if code in g.meta or code in sold or g.cooldown.get(code, -1) >= t:
            continue
        if not (px[i] > 0) or not d['valid'][t, i]:
            continue    # 当天停牌或无价格
        buys.append((code, c))
        slots -= 1

    if sells or buys:
        print('%s 候选 %d 只 | 卖出 %s | 买入 %s%s' % (
            today, len(cand), [s[0] for s in sells],
            ['%s(%d个信号)' % (b[0], b[1]['n']) for b in buys], '' if market_ok else ' | 市场过滤中'))
    for code, reason in sells:
        log_signal(today, '卖出', code, reason, g.meta[code]['pair'])
    for code, c in buys:
        pair, side, z = c['best']
        log_signal(today, '买入', code, '低估 z=%.2f，共 %d 个信号' % (z, c['n']), pair)
    if not g.trade:
        # 只出信号：按开盘价虚拟成交，用于后续产生卖出信号
        for code, _ in sells:
            del g.meta[code]
        for code, c in buys:
            g.meta[code] = {'pair': c['best'][0], 'side': c['best'][1],
                            'entry_px': px[d['col'][code]], 'entry_date': today}
        save_signals()
        return

    # 5. 下单：先卖后买，每只约 1/max_positions 资金
    col = d['col']
    total = cash + sum(v[0] * px[col[c]] for c, v in holding.items()
                       if c in col and px[col[c]] > 0)
    for code, reason in sells:
        vol, can_use = holding.get(code, (0, 0))
        i = d['col'].get(code)
        if i is not None and can_use > 0 and px[i] > 0:
            order(ContextInfo, 24, code, px[i], can_use, reason)
            cash += can_use * px[i] * (1 - g.fee_buffer)
            del g.meta[code]
    slot_value = total / g.max_positions
    for code, c in buys:
        i = d['col'][code]
        value = min(slot_value, cash)
        volume = int(value / (px[i] * (1 + g.fee_buffer)) / 100) * 100
        if volume <= 0:
            continue
        pair, side, z = c['best']
        order(ContextInfo, 23, code, px[i], volume,
              '%s 相对 %s 低估 z=%.2f，共 %d 个信号' % (code, pair['x'] if side == 'y' else pair['y'], z, c['n']))
        cash -= volume * px[i] * (1 + g.fee_buffer)
        g.meta[code] = {'pair': pair, 'side': side, 'entry_px': px[i], 'entry_date': today}
    save_signals()


# ============================================================
def get_account():
    cash, holding = 0.0, {}
    try:
        acc = get_trade_detail_data(g.acct, 'stock', 'account')
        if acc:
            cash = acc[0].m_dAvailable
        for p in get_trade_detail_data(g.acct, 'stock', 'position'):
            holding[p.m_strInstrumentID + '.' + p.m_strExchangeID] = (p.m_nVolume, p.m_nCanUseVolume)
    except Exception as e:
        print('读取账户失败: %s' % e)
    return cash, holding


def order(ContextInfo, op_type, code, price, volume, reason):
    passorder(op_type, 1101, g.acct, code, 11, price, volume, 'pair_signal', 1, '', ContextInfo)
    print('  %s %s %d 股 @ %.2f  %s' % ('买入' if op_type == 23 else '卖出', code, volume, price, reason))


def log_signal(today, action, code, reason, pair):
    g.signal_rows.append({'date': today, 'action': action, 'code': code, 'reason': reason,
                          'pair_x': pair['x'], 'pair_y': pair['y'],
                          'beta': round(pair['beta'], 4), 'pvalue': round(pair['p'], 4)})


def save_signals():
    if not g.signal_rows:
        return
    try:
        pd.DataFrame(g.signal_rows).to_csv(g.signal_file, index=False, encoding='gbk')
    except Exception:
        pass
