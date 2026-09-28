# coding: gbk
"""
配对信号策略 —— QMT 实盘交易版

选股和信号逻辑与回测版 qmt_pair_signal.py 完全相同（同样的参数得到同样的信号），
区别在于执行和安全：
  - 按时间运行：每天 g.plan_time 之后生成当天的交易计划（只用昨天及以前的数据），
    g.trade_start 之后开始下单
  - 限价委托：买入按卖一价、卖出按买一价（加少量滑点），不超过涨跌停价；
    未成交的委托每隔 g.reprice_seconds 秒撤单后按新价格重挂
  - 涨停不买、跌停不卖（挂不上也不会成交），跌停的卖单会在当天持续重试，
    收盘前没卖掉的第二天重新判断
  - 盘中止损：持仓价格跌破止损线时当天就卖出
  - 持仓信息（每只股票对应的配对、买入价、冷却期）和候选配对池保存到
    g.state_file，QMT 重启后自动恢复
  - 行情数据过期（最新日线不是上一个交易日）时拒绝交易并提示
  - 只管理本策略买入的股票，账户里手动买的股票不受影响

使用方法：
  1. 新建 Python 策略，粘贴本文件全部内容（第一行保持 # coding: gbk）
  2. 在 set_params() 里填写资金账号 g.account，设置股票池和参数
  3. 数据管理里下载股票池的日线和除权数据（至少最近 3 年），之后每天盘后补充
  4. 主图选一只交易活跃的股票（如 600000.SH），周期任意，选择「实盘」模式运行
  5. 先用 g.trade = False 运行几天：只打印交易计划、不下单，确认无误后再改为 True

注意：
  - 本文件只用于实盘/模拟盘，不能用于回测（回测请用 qmt_pair_signal.py）
  - 首次运行会重新选股；状态文件在 g.state_file，删除它等于策略「从零开始」
"""
import datetime
import json
import math
import os
import time

import numpy as np
import pandas as pd


class _G(object):
    pass


g = _G()


def set_params():
    # ---- 账户 ----
    g.account = ''                  # 资金账号（必填），如 '8888888888'
    g.account_type = 'STOCK'
    # ---- 股票池：sector 或 codes 二选一（codes 非空时优先）----
    g.sector = '沪深300'
    g.codes = []
    # ---- 选股（与回测版一致）----
    g.formation_days = 500
    g.reselect_every = 20
    g.min_corr = 0.85
    g.max_candidates = 3000
    g.max_pvalue = 0.05
    g.min_half_life = 2
    g.max_half_life = 40
    g.pool_size = 60
    g.min_amount = 5e7
    g.max_missing = 0.05
    # ---- 信号（与回测版一致）----
    g.window = 120
    g.entry_z = 2.0
    g.exit_z = 0.0
    # ---- 持仓与风控（与回测版一致）----
    g.max_positions = 5
    g.stop_loss = 0.15
    g.max_hold_days = 60
    g.break_pvalue = 0.2
    g.cooldown_days = 20
    g.market_index = None           # 如 '000300.SH'
    g.market_ma = 60
    # ---- 资金 ----
    g.capital = 0                   # 策略可用的总资金（元）；0 表示用账户总资产
    g.max_order_value = 200000      # 单笔委托金额上限（元），防止下错大单
    # ---- 执行时间 ----
    g.plan_time = '09:20:00'        # 之后生成当天交易计划
    g.trade_start = '09:30:30'      # 之后开始下单
    g.buy_deadline = '14:50:00'     # 之后不再新买
    g.sell_deadline = '14:56:00'    # 之后不再下单
    g.loop_seconds = 10             # 执行检查的最小间隔（秒）
    g.reprice_seconds = 60          # 委托多少秒未成交就撤单重挂
    g.slippage = 0.002              # 限价相对对手价的让价比例
    # ---- 开关与文件 ----
    g.trade = False                 # False：只打印交易计划，不下单（先观察几天再打开）
    g.auto_download = True          # 生成计划前自动补充最近的日线数据
    g.remark = 'pairsig'            # 委托备注，用来识别本策略的委托
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    base = desktop if os.path.isdir(desktop) else os.path.expanduser('~')
    g.state_file = os.path.join(base, 'pair_live_state.json')
    g.log_file = os.path.join(base, 'pair_live_log.csv')


# ============================================================
def init(ContextInfo):
    set_params()
    g.ready = bool(g.account)
    if not g.ready:
        print('【未启动】请在 set_params() 里填写资金账号 g.account')
        return
    ContextInfo.set_account(g.account)
    g.data = None
    g.data_date = None
    g.plan = None
    g.last_loop = 0.0
    g.placed = {}          # code -> 本策略最近一次下单的时间戳（用于撤单重挂）
    load_state()
    print('配对实盘策略已启动：账号 %s，%s，持仓 %d 只，候选配对 %d 对' % (
        g.account, '实盘下单' if g.trade else '只打印计划（g.trade=False）',
        len(g.meta), len(g.pool)))


def handlebar(ContextInfo):
    if not g.ready:
        return
    if ContextInfo.do_back_test:
        if not getattr(g, 'warned_bt', False):
            print('本文件是实盘版，不能回测；回测请用 qmt_pair_signal.py')
            g.warned_bt = True
        return
    if not ContextInfo.is_last_bar():
        return
    now = datetime.datetime.now()
    today, hms = now.strftime('%Y%m%d'), now.strftime('%H:%M:%S')
    try:
        if (g.plan is None or g.plan['date'] != today) and hms >= g.plan_time:
            make_plan(ContextInfo, today)
        if g.plan is not None and g.plan['date'] == today and g.plan.get('ok') \
                and g.trade_start <= hms <= '14:57:00' \
                and time.time() - g.last_loop >= g.loop_seconds:
            g.last_loop = time.time()
            execute(ContextInfo, today, hms)
    except Exception as e:
        import traceback
        print('%s %s 运行出错: %s' % (today, hms, e))
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
# 状态保存 / 恢复
# ============================================================
def load_state():
    g.meta, g.pool, g.cooldown, g.last_select_date = {}, [], {}, None
    if not os.path.exists(g.state_file):
        return
    try:
        with open(g.state_file, 'r') as f:
            s = json.load(f)
        g.meta = s.get('meta', {})
        g.pool = s.get('pool', [])
        g.cooldown = s.get('cooldown', {})
        g.last_select_date = s.get('last_select_date')
    except Exception as e:
        print('读取状态文件失败（%s），按空状态启动: %s' % (g.state_file, e))


def save_state():
    s = {'meta': g.meta, 'pool': g.pool, 'cooldown': g.cooldown,
         'last_select_date': g.last_select_date,
         'saved_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    tmp = g.state_file + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(s, f, ensure_ascii=True, indent=1,
                  default=lambda o: o.item() if hasattr(o, 'item') else str(o))
    if os.path.exists(g.state_file):
        os.remove(g.state_file)
    os.rename(tmp, g.state_file)


def log_event(action, code, volume, price, reason):
    row = {'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'action': action,
           'code': code, 'volume': volume, 'price': round(price, 3), 'reason': reason}
    print('  %s %s %s %d 股 @ %.2f  %s' % (row['time'][11:], action, code, volume, price, reason))
    try:
        new = not os.path.exists(g.log_file)
        pd.DataFrame([row]).to_csv(g.log_file, mode='a', header=new, index=False, encoding='gbk')
    except Exception:
        pass


# ============================================================
# 数据：每天读一次截至昨天的日线
# ============================================================
def load_data(ContextInfo, today):
    codes = list(g.codes) or ContextInfo.get_stock_list_in_sector(g.sector)
    if not codes:
        print('股票池为空：请检查板块名称 %s 或填写 g.codes' % g.sector)
        return None
    days = int((g.formation_days + g.window + 20) * 1.6)
    start = (pd.Timestamp(today) - pd.Timedelta(days=days)).strftime('%Y%m%d')
    if g.auto_download:
        t0 = time.time()
        recent = (pd.Timestamp(today) - pd.Timedelta(days=15)).strftime('%Y%m%d')
        for c in codes + ([g.market_index] if g.market_index else []):
            try:
                download_history_data(c, '1d', recent, '')
            except Exception:
                pass
        print('补充最近日线数据，用时 %.0f 秒' % (time.time() - t0))
    t0 = time.time()

    def wide(fields, dividend_type, cs):
        res = ContextInfo.get_market_data_ex(fields, cs, period='1d', start_time=start,
                                             end_time=today, count=-1,
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
                    cols[c] = s[[i < today for i in s.index]]    # 只要今天以前的
            out[f] = pd.DataFrame(cols).sort_index()
        return out

    close = wide(['close'], 'front_ratio', codes)['close']
    if close.empty:
        print('没有读到行情，请先在「数据管理」下载股票池日线和除权数据')
        return None
    codes, dates = list(close.columns), list(close.index)
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
    print('读取 %d 只股票、%d 个交易日行情（截至 %s），用时 %.1f 秒'
          % (len(codes), len(dates), dates[-1], time.time() - t0))
    return {
        'codes': codes, 'col': dict((c, i) for i, c in enumerate(codes)),
        'dates': dates, 'idx': dict((d, i) for i, d in enumerate(dates)),
        'logc': np.log(close.ffill().values.astype(float)),
        'close': close.ffill().values.astype(float),      # 前复权收盘价（算止损用比值，复权方式不影响）
        'valid': close.notna().values & (vol.fillna(0).values > 0),
        'amount': amt.values.astype(float),
        'st': st, 'mkt': mkt,
    }


def previous_trading_day(ContextInfo, today):
    """上一个交易日；取不到时返回 None"""
    for code in ('SH', '000001.SH'):
        try:
            ds = ContextInfo.get_trading_dates(code, '', today, 5, '1d')
            ds = [str(x)[:8] for x in ds if str(x)[:8] < today]
            if ds:
                return max(ds)
        except Exception:
            pass
    print('  提示：取不到交易日历，无法检查行情是否最新，请自行确认日线数据已更新到上一交易日')
    return None


def entry_return(code, m, last_ratio=1.0):
    """买入以来的收益：用同一条前复权序列里「买入前一天」和「最近一天」的收盘价比，
    分红除权不影响。last_ratio = 今天最新价 / 昨收（盘中用）"""
    d = g.data
    i = d['col'].get(code)
    k = int(np.searchsorted(d['dates'], m['entry_date'])) - 1
    if i is None or k < 0 or not d['close'][k, i] > 0:
        return 0.0
    return d['close'][-1, i] * last_ratio / d['close'][k, i] - 1


def remap():
    col = g.data['col']
    for pair in g.pool + [m['pair'] for m in g.meta.values()]:
        pair['ix'], pair['iy'] = col.get(pair['x'], -1), col.get(pair['y'], -1)
    g.pool = [pr for pr in g.pool if pr['ix'] >= 0 and pr['iy'] >= 0]


def days_between(d1, t):
    """日期 d1 到第 t 个交易日之间的交易日数（t = 今天以前的交易日数）"""
    if d1 is None:
        return 10 ** 6
    return t - int(np.searchsorted(g.data['dates'], d1))


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
# 每天生成交易计划（只用昨天及以前的数据，与回测版逻辑相同）
# ============================================================
def make_plan(ContextInfo, today):
    g.plan = {'date': today, 'ok': False, 'sells': {}, 'buys': {}}
    g.data = load_data(ContextInfo, today)
    if g.data is None:
        return
    d = g.data
    remap()
    prev = previous_trading_day(ContextInfo, today)
    if prev is not None and d['dates'][-1] < prev:
        print('【今日不交易】日线数据只到 %s，上一个交易日是 %s。请在「数据管理」补充数据后重启策略'
              % (d['dates'][-1], prev))
        return
    t = len(d['dates'])          # 今天在数据里的位置（数据只含今天以前）
    if t < max(g.formation_days, g.window):
        print('【今日不交易】历史数据只有 %d 天，至少需要 %d 天' % (t, max(g.formation_days, g.window)))
        return

    # 1. 定期重新选股；检查持仓配对关系
    broken = set()
    if days_between(g.last_select_date, t) >= g.reselect_every:
        t0 = time.time()
        g.pool = select_pairs(t)
        g.last_select_date = today
        for code, m in g.meta.items():
            if m['pair'].get('ix', -1) < 0 or pair_pvalue(m['pair'], t) > g.break_pvalue:
                broken.add(code)
        print('%s 重新选股：候选配对 %d 对，用时 %.1f 秒' % (today, len(g.pool), time.time() - t0))

    # 2. 与账户核对：本策略记录的持仓如果账户里已经没有了，删除记录
    cash, total, holding = get_account(ContextInfo)
    if g.meta and not holding:
        print('【今日不交易】账户持仓为空，但策略记录有 %d 只持仓，可能是账户数据还没同步。'
              '确认后重启策略；如果确实已全部卖出，删除状态文件 %s' % (len(g.meta), g.state_file))
        return
    for code in list(g.meta.keys()):
        if holding.get(code, {}).get('volume', 0) <= 0:
            print('  %s 账户中已无持仓，删除策略记录' % code)
            del g.meta[code]

    # 3. 卖出计划
    for code, m in g.meta.items():
        i = d['col'].get(code)
        if i is None or m['pair'].get('ix', -1) < 0:
            if code in broken:
                g.plan['sells'][code] = '已不在股票池'
            continue
        z = zscore(m['pair'], t)
        ret = entry_return(code, m)
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
            g.plan['sells'][code] = reason

    # 4. 买入计划
    market_ok = True
    if d['mkt'] is not None and t > g.market_ma:
        mk = d['mkt'][t - g.market_ma:t]
        market_ok = bool(np.isfinite(mk).all() and mk[-1] >= mk.mean())
    cand = {}
    for pair in g.pool:
        z = zscore(pair, t)
        if z is None or abs(z) < g.entry_z:
            continue
        side = 'y' if z < 0 else 'x'
        code = pair[side]
        c = cand.setdefault(code, {'n': 0, 'score': 0.0, 'best': None})
        c['n'] += 1
        c['score'] += abs(z)
        if c['best'] is None or abs(z) > abs(c['best'][2]):
            c['best'] = (pair, side, z)
    ranked = sorted(cand.items(), key=lambda kv: (-kv[1]['n'], -kv[1]['score']))
    slots = g.max_positions - (len(g.meta) - len(g.plan['sells']))
    capital = g.capital if g.capital > 0 else total
    for code, c in ranked:
        if slots <= 0 or not market_ok:
            break
        cd = g.cooldown.get(code)
        if code in g.meta or code in g.plan['sells'] or \
                (cd and days_between(cd[0], t) < cd[1]):
            continue
        if holding.get(code, {}).get('volume', 0) > 0:
            continue    # 账户里已有这只股票（可能是手动持仓），不碰
        pair, side, z = c['best']
        g.plan['buys'][code] = {
            'value': capital / g.max_positions,
            'pair': pair, 'side': side,
            'reason': '相对 %s 低估 z=%.2f，共 %d 个信号' % (
                pair['x'] if side == 'y' else pair['y'], z, c['n'])}
        slots -= 1

    g.plan['ok'] = True
    save_state()
    print('===== %s 交易计划（%s）=====' % (today, '将自动下单' if g.trade else '只打印，不下单'))
    for code, reason in g.plan['sells'].items():
        print('  卖出 %s：%s' % (code, reason))
    for code, b in g.plan['buys'].items():
        print('  买入 %s：约 %.0f 元，%s' % (code, b['value'], b['reason']))
    if not g.plan['sells'] and not g.plan['buys']:
        print('  今日无交易')
    if not market_ok:
        print('  市场过滤中：指数低于 %d 日均线，不开新仓' % g.market_ma)


# ============================================================
# 盘中执行：限价委托、撤单重挂、盘中止损
# ============================================================
ACTIVE = (48, 49, 50, 51, 52, 55)     # 未报/待报/已报/已报待撤/部成待撤/部成


def execute(ContextInfo, today, hms):
    cash, total, holding = get_account(ContextInfo)
    active = active_orders()
    codes = list(set(list(g.plan['sells'].keys()) + list(g.plan['buys'].keys()) +
                     list(g.meta.keys())))
    if not codes:
        return
    ticks = ContextInfo.get_full_tick(codes)

    # 盘中止损
    for code, m in g.meta.items():
        tk = ticks.get(code) or {}
        last, pre = tk.get('lastPrice', 0), tk.get('lastClose', 0)
        if code in g.plan['sells'] or m['entry_date'] == today or not (last > 0 and pre > 0):
            continue
        ret = entry_return(code, m, last / pre)
        if ret <= -g.stop_loss:
            g.plan['sells'][code] = '盘中止损 %.1f%%' % (ret * 100)
            print('  %s 触发盘中止损 %.1f%%' % (code, ret * 100))

    # 撤掉超时未成交的委托，下一轮按新价格重挂；收盘前撤掉全部未成交委托
    for code, orders in active.items():
        if time.time() - g.placed.get(code, 0) >= g.reprice_seconds or hms > g.sell_deadline:
            for o in orders:
                cancel_order(ContextInfo, o)
    if hms > g.sell_deadline:
        return
    # 刚下过单的股票，委托查询可能还没更新：等 reprice_seconds 后再判断，避免重复下单
    recent = set(c for c, ts in g.placed.items() if time.time() - ts < g.reprice_seconds)

    # 卖出
    for code, reason in list(g.plan['sells'].items()):
        pos = holding.get(code, {})
        if pos.get('volume', 0) <= 0:
            finish_sell(code, reason, today)
            continue
        if code in active or code in recent:
            continue
        can_use = pos.get('can_use', 0)
        tk = ticks.get(code) or {}
        bid = first(tk.get('bidPrice'))
        low_limit = limit_price(ContextInfo, code, tk, up=False)
        if can_use <= 0 or bid <= 0 or (low_limit and bid <= low_limit + 1e-6):
            continue    # 今天买的不能卖 / 跌停无买盘，下一轮再试
        price = round_price(max(bid * (1 - g.slippage), low_limit or 0))
        place(ContextInfo, 24, code, price, can_use, reason)

    # 买入
    if hms > g.buy_deadline:
        return
    for code, b in list(g.plan['buys'].items()):
        if code in active or code in recent or b.get('done'):
            continue
        tk = ticks.get(code) or {}
        ask = first(tk.get('askPrice'))
        up_limit = limit_price(ContextInfo, code, tk, up=True)
        if ask <= 0 or (up_limit and ask >= up_limit - 1e-6):
            continue    # 涨停或停牌，下一轮再试
        price = round_price(min(ask * (1 + g.slippage), up_limit or 1e12))
        have = holding.get(code, {}).get('volume', 0)
        if have > 0 and code not in g.meta:
            g.meta[code] = {'pair': b['pair'], 'side': b['side'], 'entry_date': today}
            save_state()
        # 第一次下单时按当时价格定下目标股数，之后只补未成交的部分
        if 'target' not in b:
            b['target'] = int(b['value'] / price / 100) * 100
        want = b['target'] - have
        volume = int(min(want * price, cash * 0.998, g.max_order_value) / price / 100) * 100
        if volume < 100:
            if want < 100:
                b['done'] = True
            continue
        place(ContextInfo, 23, code, price, volume, b['reason'])
        cash -= volume * price


def finish_sell(code, reason, today):
    """确认卖完：删除持仓记录，止损/超时/失效的设置冷却期"""
    if code in g.meta:
        del g.meta[code]
        if not reason.startswith('价差回归'):
            g.cooldown[code] = [today, g.cooldown_days]
        save_state()
        print('  %s 已全部卖出（%s）' % (code, reason))
    del g.plan['sells'][code]


def place(ContextInfo, op, code, price, volume, reason):
    action = '买入' if op == 23 else '卖出'
    if not g.trade:
        if not g.plan.setdefault('printed', {}).get(code):
            log_event('计划' + action + '（未下单）', code, volume, price, reason)
            g.plan['printed'][code] = True
        return
    passorder(op, 1101, g.account, code, 11, price, volume, 'pair_live', 2,
              g.remark, ContextInfo)
    g.placed[code] = time.time()
    log_event(action, code, volume, price, reason)


def cancel_order(ContextInfo, o):
    if not g.trade:
        return
    try:
        cancel(o.m_strOrderSysID, g.account, g.account_type, ContextInfo)
    except Exception as e:
        print('  撤单失败 %s: %s' % (o.m_strOrderSysID, e))


def active_orders():
    """本策略未完成的委托：code -> [order]"""
    out = {}
    try:
        for o in get_trade_detail_data(g.account, 'stock', 'order'):
            if str(getattr(o, 'm_strRemark', '')).startswith(g.remark) and \
                    o.m_nOrderStatus in ACTIVE:
                out.setdefault(o.m_strInstrumentID + '.' + o.m_strExchangeID, []).append(o)
    except Exception as e:
        print('  查询委托失败: %s' % e)
    return out


def get_account(ContextInfo):
    """返回 (可用资金, 总资产, {code: {volume, can_use}})"""
    cash, total, holding = 0.0, 0.0, {}
    acc = get_trade_detail_data(g.account, 'stock', 'account')
    if acc:
        cash, total = acc[0].m_dAvailable, acc[0].m_dBalance
    for p in get_trade_detail_data(g.account, 'stock', 'position'):
        holding[p.m_strInstrumentID + '.' + p.m_strExchangeID] = {
            'volume': p.m_nVolume, 'can_use': p.m_nCanUseVolume}
    return cash, total, holding


def first(x):
    try:
        return float(x[0]) if isinstance(x, (list, tuple)) else float(x or 0)
    except Exception:
        return 0.0


def round_price(p):
    return math.floor(p * 100 + 0.5) / 100.0


def limit_price(ContextInfo, code, tk, up):
    """涨跌停价：优先取合约信息，取不到按板块规则用昨收估算"""
    try:
        det = ContextInfo.get_instrumentdetail(code)
        v = float(det.get('UpStopPrice' if up else 'DownStopPrice', 0) or 0)
        if v > 0:
            return v
    except Exception:
        pass
    pre = float(tk.get('lastClose', 0) or 0)
    if pre <= 0:
        return 0.0
    pct = 0.2 if code[:3] in ('300', '301', '688', '689') else 0.1
    return round_price(pre * (1 + pct if up else 1 - pct))
