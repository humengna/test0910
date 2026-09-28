# coding: gbk
"""
一对多（中心股）信号策略 —— QMT 版（动态选股，不固定配对）

思路：
  - 定期（默认每 20 个交易日）用过去约 2 年的数据，在股票池里做两两协整检验，
    找出同时和至少 3 只股票协整的「中心股」，每个中心股取最多 5 只伙伴股
  - 每个交易日，对每个中心股计算它和每只伙伴股的价差 z 值，取平均得到「综合 z」：
      综合 z <= -1.5，且至少 2 对的 z <= -1.0  -> 中心股相对整个篮子被低估，买入中心股
      综合 z >=  1.5，且至少 2 对的 z >=  1.0  -> 中心股相对篮子偏贵，买入其中
                                                   z 最大（相对中心股最便宜）的伙伴股
    只有 1 对发出信号时不交易（多半是那只伙伴股自己的消息）
  - 同一只股票可能被多个中心股同时选中：信号越多、偏离越大，排名越靠前
  - 每只股票单独持仓、单独退出：综合 z 回到 0、止损、超过最长持有天数、
    或重新选股时仍协整的伙伴不足 2 只（关系失效）时卖出
  - 只用当天之前的数据，回测天然是「滚动样本外」，没有未来数据

价差用对数价格：spread_i = ln(中心股) - beta_i * ln(伙伴股_i)。

使用方法：
  1. 新建 Python 策略，粘贴本文件全部内容（第一行保持 # coding: gbk）
  2. 修改 set_params() 里的股票池和参数
  3. 数据管理里下载股票池的日线和除权数据，起始时间比回测开始早约 3 年
  4. 回测：周期选「日线」，设置回测区间、资金、费率。实盘/模拟：同样日线运行
  5. g.trade = False 时只输出信号不下单（每日信号保存在 g.signal_file）
"""
import math
import os
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
    # ---- 选股（定期重新做协整检验，找出中心股）----
    g.formation_days = 500          # 选股用的历史长度（交易日，约 2 年）
    g.reselect_every = 20           # 每隔多少个交易日重新选股（1 = 每天，很慢）
    g.min_corr = 0.85               # 对数价格相关系数下限（初筛）
    g.max_candidates = 3000         # 初筛后最多检验多少对（按相关系数取前 N，控制速度）
    g.max_pvalue = 0.05             # 协整 p 值上限
    g.min_half_life = 2             # 半衰期范围（交易日）
    g.max_half_life = 40
    g.min_amount = 5e7              # 选股期日均成交额下限（元）
    g.max_missing = 0.05            # 选股期允许的缺失/停牌比例
    g.min_partners = 3              # 和至少几只股票协整才算中心股
    g.max_partners = 5              # 每个中心股最多用几只伙伴股（按 p 值取最好的）
    g.hub_pool_size = 30            # 候选中心股最多保留多少个
    # ---- 信号 ----
    g.window = 120                  # 计算 z 值的窗口（交易日）
    g.entry_z = 1.5                 # 综合 z 的绝对值超过它才考虑买入
    g.agree_z = 1.0                 # 单对 z 的绝对值超过它，算一个同方向信号
    g.min_agree = 2                 # 至少几对同方向才买入
    g.exit_z = 0.0                  # 综合 z 回到它（穿越均值）就卖出
    # ---- 持仓与风控 ----
    g.max_positions = 5             # 最多同时持有几只，每只约 1/N 资金
    g.stop_loss = 0.15              # 从买入价下跌超过 15% 止损
    g.max_hold_days = 60            # 最长持有天数（交易日）
    g.break_pvalue = 0.2            # 重新选股时，伙伴对 p 值低于它才算仍然协整
    g.cooldown_days = 20            # 止损/超时/关系失效卖出后，这只股票多少天内不再买入
    # 市场过滤：指数收盘价低于 N 日均线时不开新仓（None 表示不启用）
    g.market_index = None           # 如 '000300.SH'
    g.market_ma = 60
    # ---- 执行 ----
    g.trade = True                  # False：只输出信号，不下单
    g.fee_buffer = 0.0005
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    base = desktop if os.path.isdir(desktop) else os.path.expanduser('~')
    g.signal_file = os.path.join(base, 'hub_signals.csv')


# ============================================================
def init(ContextInfo):
    set_params()
    g.acct = getattr(ContextInfo, 'accID', '') or 'test'
    try:
        ContextInfo.set_account(g.acct)
    except Exception:
        pass
    g.data = None          # 行情缓存
    g.load_start = None    # 读取行情的起始日期
    g.hubs = []            # 候选中心股：dict(h, ih, partners=[dict(code, i, beta, p)], avg_p)
    g.last_select_date = None
    g.meta = {}            # 持仓信息：code -> dict(hub, side, partner, entry_px, entry_date)
    g.cooldown = {}        # code -> 该日期序号（含）之前不再买入
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
def load_data(ContextInfo, start, end):
    """一次性读取股票池全部历史，之后按日期序号切片（只用当天之前的行）"""
    codes = list(g.codes) or ContextInfo.get_stock_list_in_sector(g.sector)
    if not codes:
        print('股票池为空：请检查板块名称 %s 或填写 g.codes' % g.sector)
        return None
    t0 = time.time()

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
        if g.load_start is None:
            # 只读需要的历史：第一次运行日（回测开始日）往前 选股期 + z 窗口，按自然日放宽
            days = int((g.formation_days + g.window + 20) * 1.6)
            g.load_start = (pd.Timestamp(today) - pd.Timedelta(days=days)).strftime('%Y%m%d')
        g.data = load_data(ContextInfo, g.load_start, end)
        if g.data is not None:
            remap()
    return g.data is not None and today in g.data['idx']


def remap():
    """重新读数据后股票列顺序可能变化，按代码重新定位中心股和伙伴股的列号"""
    col = g.data['col']
    for hub in g.hubs + [m['hub'] for m in g.meta.values()]:
        hub['ih'] = col.get(hub['h'], -1)
        for pt in hub['partners']:
            pt['i'] = col.get(pt['code'], -1)
    g.hubs = [hb for hb in g.hubs if hub_ok(hb)]


def hub_ok(hub):
    return hub['ih'] >= 0 and all(pt['i'] >= 0 for pt in hub['partners'])


def days_between(d1, t):
    """日期 d1 到第 t 个交易日之间的交易日数"""
    i = g.data['idx'].get(d1)
    return t - i if i is not None else 10 ** 6


# ============================================================
# 选股：在 [t-formation, t) 上两两做协整检验，找出中心股
# ============================================================
def scan_pairs(t):
    """返回所有通过协整、半衰期检验的配对 (a, b, p)，a、b 为列号"""
    d = g.data
    lo = t - g.formation_days
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
    pairs = []
    for k in sel:
        a, b = idx[iu[0][k]], idx[iu[1][k]]
        best = None
        for ix, iy in ((a, b), (b, a)):
            try:
                p, beta, resid = coint_test(L[:, iy], L[:, ix])
            except Exception:
                continue
            if best is None or p < best[0]:
                best = (p, beta, resid)
        if best is None:
            continue
        p, beta, resid = best
        if p > g.max_pvalue or beta <= 0:
            continue
        if g.min_half_life <= half_life(resid) <= g.max_half_life:
            pairs.append((a, b, p))
    return pairs


def select_hubs(t):
    d = g.data
    L = d['logc'][t - g.formation_days:t]
    adj = {}
    for a, b, p in scan_pairs(t):
        adj.setdefault(a, []).append((p, b))
        adj.setdefault(b, []).append((p, a))
    hubs = []
    for ih, lst in adj.items():
        if len(lst) < g.min_partners:
            continue
        partners = []
        for p, ip in sorted(lst)[:g.max_partners]:
            # 以中心股为因变量估计 beta：ln(中心股) = a + beta * ln(伙伴股)
            x = L[:, ip]
            beta = _ols(L[:, ih], np.column_stack([np.ones_like(x), x]))[0][1]
            if beta > 0:
                partners.append({'code': d['codes'][ip], 'i': ip, 'beta': beta, 'p': p})
        if len(partners) >= g.min_partners:
            hubs.append({'h': d['codes'][ih], 'ih': ih, 'partners': partners,
                         'avg_p': float(np.mean([pt['p'] for pt in partners]))})
    hubs.sort(key=lambda hb: (-len(hb['partners']), hb['avg_p']))
    return hubs[:g.hub_pool_size]


def partners_still_coint(hub, t):
    """重新检验中心股与每只伙伴股，返回仍然协整（p < break_pvalue）的伙伴数"""
    if not hub_ok(hub):
        return 0
    L = g.data['logc'][t - g.formation_days:t]
    n = 0
    for pt in hub['partners']:
        try:
            if coint_test(L[:, hub['ih']], L[:, pt['i']])[0] < g.break_pvalue:
                n += 1
        except Exception:
            pass
    return n


def hub_z(hub, t):
    """每只伙伴股一个 z 值（窗口 [t-window, t)，不含当天），无效为 NaN"""
    L = g.data['logc'][t - g.window:t]
    zs = np.full(len(hub['partners']), np.nan)
    if hub['ih'] < 0:
        return zs
    for k, pt in enumerate(hub['partners']):
        if pt['i'] < 0:
            continue
        s = L[:, hub['ih']] - pt['beta'] * L[:, pt['i']]
        sd = s.std()
        if np.isfinite(s).all() and sd > 0:
            zs[k] = (s[-1] - s.mean()) / sd
    return zs


def composite(zs):
    """返回 (综合 z, 有效对数)；有效对数不足 min_agree 时综合 z 为 None"""
    ok = zs[np.isfinite(zs)]
    return (float(ok.mean()) if len(ok) >= g.min_agree else None), len(ok)


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

    # 1. 定期重新选股；同时检查持仓对应中心股的关系是否失效
    broken = set()
    if g.last_select_date is None or days_between(g.last_select_date, t) >= g.reselect_every:
        t0 = time.time()
        g.hubs = select_hubs(t)
        g.last_select_date = today
        for code, m in g.meta.items():
            if partners_still_coint(m['hub'], t) < g.min_agree:
                broken.add(code)
        print('%s 重新选股：中心股 %d 个，用时 %.1f 秒' % (today, len(g.hubs), time.time() - t0))

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
        if i is None:
            if code in broken:
                sells.append((code, '已不在股票池'))
            continue
        zbar, _ = composite(hub_z(m['hub'], t))
        ret = px[i] / m['entry_px'] - 1 if px[i] > 0 else 0.0
        reason = None
        if code in broken:
            reason = '中心股关系失效'
        elif ret <= -g.stop_loss:
            reason = '止损 %.1f%%' % (ret * 100)
        elif days_between(m['entry_date'], t) >= g.max_hold_days:
            reason = '持有超过 %d 天' % g.max_hold_days
        elif zbar is not None and ((m['side'] == 'hub' and zbar >= -g.exit_z) or
                                   (m['side'] == 'partner' and zbar <= g.exit_z)):
            reason = '综合z回归 %.2f' % zbar
        if reason:
            sells.append((code, reason))
            if not reason.startswith('综合z回归'):
                g.cooldown[code] = t + g.cooldown_days - 1

    # 4. 买入信号：综合 z 超过阈值且至少 min_agree 对同方向
    market_ok = True
    if d['mkt'] is not None and t > g.market_ma:
        mk = d['mkt'][t - g.market_ma:t]
        market_ok = bool(np.isfinite(mk).all() and mk[-1] >= mk.mean())
    cand = {}
    for hub in g.hubs:
        zs = hub_z(hub, t)
        zbar, nvalid = composite(zs)
        if zbar is None:
            continue
        if zbar <= -g.entry_z and np.sum(zs <= -g.agree_z) >= g.min_agree:
            side, code, agree = 'hub', hub['h'], int(np.sum(zs <= -g.agree_z))
            partner = None
        elif zbar >= g.entry_z and np.sum(zs >= g.agree_z) >= g.min_agree:
            k = int(np.nanargmax(zs))
            side, partner = 'partner', hub['partners'][k]['code']
            code, agree = partner, int(np.sum(zs >= g.agree_z))
        else:
            continue
        c = cand.setdefault(code, {'n': 0, 'score': 0.0, 'best': None})
        c['n'] += 1
        c['score'] += abs(zbar)
        if c['best'] is None or abs(zbar) > abs(c['best']['zbar']):
            c['best'] = {'hub': hub, 'side': side, 'partner': partner, 'zbar': zbar,
                         'agree': agree, 'nvalid': nvalid}
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
            today, len(cand), [s[0] for s in sells], [b[0] for b in buys],
            '' if market_ok else ' | 市场过滤中'))
    for code, reason in sells:
        log_signal(today, '卖出', code, reason, g.meta[code])
    for code, c in buys:
        log_signal(today, '买入', code, buy_reason(c), c['best'])
    if not g.trade:
        # 只出信号：按开盘价虚拟成交，用于后续产生卖出信号
        for code, _ in sells:
            del g.meta[code]
        for code, c in buys:
            g.meta[code] = new_meta(c['best'], px[d['col'][code]], today)
        save_signals()
        return

    # 5. 下单：先卖后买，每只约 1/max_positions 资金
    col = d['col']
    total = cash + sum(v[0] * px[col[c]] for c, v in holding.items()
                       if c in col and px[col[c]] > 0)
    for code, reason in sells:
        vol, can_use = holding.get(code, (0, 0))
        i = col.get(code)
        if i is not None and can_use > 0 and px[i] > 0:
            order(ContextInfo, 24, code, px[i], can_use, reason)
            cash += can_use * px[i] * (1 - g.fee_buffer)
            del g.meta[code]
    slot_value = total / g.max_positions
    for code, c in buys:
        i = col[code]
        value = min(slot_value, cash)
        volume = int(value / (px[i] * (1 + g.fee_buffer)) / 100) * 100
        if volume <= 0:
            continue
        order(ContextInfo, 23, code, px[i], volume, buy_reason(c))
        cash -= volume * px[i] * (1 + g.fee_buffer)
        g.meta[code] = new_meta(c['best'], px[i], today)
    save_signals()


def new_meta(best, entry_px, today):
    return {'hub': best['hub'], 'side': best['side'], 'partner': best['partner'],
            'entry_px': entry_px, 'entry_date': today}


def buy_reason(c):
    b = c['best']
    what = '中心股低估' if b['side'] == 'hub' else '相对中心股 %s 低估' % b['hub']['h']
    return '%s 综合z=%.2f，%d/%d 对同向，共 %d 个中心股信号' % (
        what, b['zbar'], b['agree'], b['nvalid'], c['n'])


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
    passorder(op_type, 1101, g.acct, code, 11, price, volume, 'hub_signal', 1, '', ContextInfo)
    print('  %s %s %d 股 @ %.2f  %s' % ('买入' if op_type == 23 else '卖出', code, volume, price, reason))


def log_signal(today, action, code, reason, info):
    hub = info['hub']
    g.signal_rows.append({'date': today, 'action': action, 'code': code, 'reason': reason,
                          'hub': hub['h'], 'side': info['side'],
                          'partners': ','.join(pt['code'] for pt in hub['partners']),
                          'betas': ','.join('%.4f' % pt['beta'] for pt in hub['partners'])})


def save_signals():
    if not g.signal_rows:
        return
    try:
        pd.DataFrame(g.signal_rows).to_csv(g.signal_file, index=False, encoding='gbk')
    except Exception:
        pass
