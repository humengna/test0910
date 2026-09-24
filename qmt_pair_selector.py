# coding: gbk
"""
配对选股 —— QMT 内置 Python 版（在 QMT 策略编辑器里直接运行）

与 pair_selector.py 功能相同，区别：
  - 不依赖 statsmodels / argparse / xtquant，只用 QMT 内置的 numpy、pandas
    （协整检验用 numpy 实现 Engle-Granger + MacKinnon p 值，结果与 statsmodels.coint 一致）
  - 参数在 set_params() 里修改
  - 数据通过 ContextInfo.get_market_data_ex 读取本地行情

需要的数据（先在 QMT「数据管理」里下载）：
  - 股票池内所有股票的「日线」行情，时间覆盖 start 往前约半年 到 oos_end（或今天）
  - 「除权数据」（用于前复权）
  用到的字段：开盘价、收盘价（前复权）、收盘价（不复权）、成交量、成交额、股票名称（判断 ST）

使用方法：
  1. 新建 Python 策略，粘贴本文件全部内容（第一行保持 # coding: gbk）
  2. 修改 set_params() 里的股票池和日期
  3. 主图任选一只股票、周期选「日线」，点「运行」（不需要回测）
  4. 结果打印在日志里，并保存到 g.out_file
"""
import math
import os
import time
import itertools

import numpy as np
import pandas as pd


class _G(object):
    pass


g = _G()


def set_params():
    # ---- 股票池：sector 或 codes 二选一（codes 非空时优先）----
    g.sector = '银行'                 # QMT 板块名，如 '银行'、'沪深300'、'上证50'
    g.codes = []                      # 如 ['600036.SH', '601166.SH', '600000.SH']
    # ---- 日期 ----
    g.start = '20200101'              # 样本内（选股）开始
    g.end = '20231231'                # 样本内结束
    g.oos_start = '20240101'          # 样本外开始，'' 表示不做样本外验证
    g.oos_end = ''                    # 样本外结束，'' 表示到最新
    # ---- 与交易策略一致的参数 ----
    g.window = 120                    # z-score 窗口，同 qmt_pair_trading.py 的 g.test_days
    g.fee = 0.001                     # 模拟回测单边换手费率
    # ---- 筛选条件 ----
    g.min_corr = 0.8                  # 对数价格最低相关系数
    g.max_pvalue = 0.05               # 协整检验最大 p 值
    g.min_half_life = 2               # 半衰期范围（交易日）
    g.max_half_life = 60
    g.min_amount = 5e7                # 样本内日均成交额下限（元）
    g.max_missing = 0.05              # 允许的缺失/停牌比例
    g.stable_parts = 3                # 稳定性检验：样本内分成几段分别做协整检验
    g.min_stable = 2                  # 至少几段协整（p<0.1）才保留，0 表示不过滤
    g.top = 20                        # 日志里打印前 N 个
    # 下载数据：True 时运行前自动补下载日线（股票多时较慢，已下载可改 False）
    g.download = False
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    g.out_file = os.path.join(desktop if os.path.isdir(desktop) else os.path.expanduser('~'),
                              'pair_candidates.csv')


def init(ContextInfo):
    set_params()
    g.done = False


def handlebar(ContextInfo):
    # 只在最后一根 K 线运行一次
    if g.done or not ContextInfo.is_last_bar():
        return
    g.done = True
    try:
        run(ContextInfo)
    except Exception as e:
        import traceback
        print('选股出错: %s' % e)
        print(traceback.format_exc())


# ============================================================
# 协整检验（numpy 实现，等价于 statsmodels.tsa.stattools.coint(y, x)）
# ============================================================
def _ols(y, X):
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X.dot(coef)
    return coef, resid


def _lagmat(x, maxlag):
    """返回矩阵，第 j 列为 x 滞后 j 期（j=0..maxlag），去掉前 maxlag 行"""
    n = len(x)
    return np.column_stack([x[maxlag - j:n - j] for j in range(maxlag + 1)])


def adf_tstat(u):
    """无常数项 ADF 检验，AIC 自动选滞后阶数，返回 t 统计量"""
    u = np.asarray(u, dtype=float)
    nobs = len(u)
    maxlag = int(math.ceil(12.0 * (nobs / 100.0) ** 0.25))
    maxlag = max(0, min(nobs // 2 - 1, maxlag))
    du = np.diff(u)

    def design(lag):
        dl = _lagmat(du, lag)                  # du_t, du_{t-1}, ..., du_{t-lag}
        n = dl.shape[0]
        y = dl[:, 0]
        X = np.column_stack([u[-n - 1:-1]] + [dl[:, j] for j in range(1, lag + 1)])
        return y, X

    # AIC 选阶（所有阶数用同一样本）。各阶模型是嵌套的，对 X'X 做一次
    # Cholesky 分解即可得到每一阶的残差平方和，不必逐阶回归
    y_full, X_full = design(maxlag)
    n = len(y_full)
    L = np.linalg.cholesky(X_full.T.dot(X_full))
    qy = np.linalg.solve(L, X_full.T.dot(y_full))
    ssr = np.maximum(y_full.dot(y_full) - np.cumsum(qy ** 2), 1e-300)
    k = np.arange(1, maxlag + 2)
    aic = n * (math.log(2 * math.pi) + np.log(ssr / n) + 1) + 2 * k
    best_lag = int(np.argmin(aic))
    # 用选定阶数重新回归
    y, X = design(best_lag)
    coef, r = _ols(y, X)
    dof = len(y) - X.shape[1]
    sigma2 = r.dot(r) / dof
    cov = sigma2 * np.linalg.inv(X.T.dot(X))
    return coef[0] / math.sqrt(cov[0, 0])


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def mackinnon_p(tstat):
    """MacKinnon (1994) 近似 p 值，2 个变量、含常数项（与 statsmodels 相同）"""
    if tstat > 0.92:
        return 1.0
    if tstat < -18.86:
        return 0.0
    if tstat <= -2.62:
        c = [2.92, 1.5012, 0.039796]
    else:
        c = [2.1945, 0.64695, -0.29198, -0.042377]
    return _norm_cdf(sum(ci * tstat ** i for i, ci in enumerate(c)))


def coint_test(y, x):
    """Engle-Granger 协整检验：返回 (p 值, alpha, beta, 残差)"""
    X = np.column_stack([np.ones_like(x), x])
    coef, resid = _ols(y, X)
    p = mackinnon_p(adf_tstat(resid))
    return p, coef[0], coef[1], resid


def half_life(resid):
    X = np.column_stack([np.ones(len(resid) - 1), resid[:-1]])
    coef, _ = _ols(np.diff(resid), X)
    lam = coef[1]
    return -math.log(2) / lam if lam < 0 else np.inf


# ============================================================
# 模拟 qmt_pair_trading.py 的交易规则
# ============================================================
def dynamic_scale(adj, raw):
    if raw is None or not raw[-1] > 0 or not adj[-1] > 0:
        return adj
    return adj * (raw[-1] / adj[-1])


def _cumsum(x):
    """前缀和（缺失值按 0 计），以及缺失值个数的前缀和"""
    ok = np.isfinite(x)
    return (np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))]),
            np.concatenate([[0], np.cumsum(~ok)]))


def rolling_z(c1, c2, r1, r2, beta, start_i, end_i, window):
    """向量化计算每天的 z-score（窗口 [t-window, t-1]，动态前复权缩放），
    与逐日循环 dynamic_scale + std 的结果相同"""
    t = np.arange(start_i, end_i)
    lo, hi = t - window, t
    # 去均值后再累加，减少大数相减的精度损失
    m1, m2 = np.nanmean(c1), np.nanmean(c2)
    a, b = c1 - m1, c2 - m2

    def wsum(x):
        cs, bad = _cumsum(x)
        return cs[hi] - cs[lo], bad[hi] - bad[lo]

    sa, na = wsum(a)
    sb, nb = wsum(b)
    saa, _ = wsum(a * a)
    sbb, _ = wsum(b * b)
    sab, _ = wsum(a * b)
    ea, eb = sa / window, sb / window
    var_a = saa / window - ea ** 2
    var_b = sbb / window - eb ** 2
    cov = sab / window - ea * eb
    # 每个窗口的缩放系数：最后一天真实价 / 前复权价（无效时为 1）
    last1, last2 = c1[hi - 1], c2[hi - 1]
    k1 = r1[hi - 1] / last1
    k2 = r2[hi - 1] / last2
    k1 = np.where((r1[hi - 1] > 0) & (last1 > 0), k1, 1.0)
    k2 = np.where((r2[hi - 1] > 0) & (last2 > 0), k2, 1.0)
    mean = k2 * (eb + m2) - beta * k1 * (ea + m1)
    var = k2 ** 2 * var_b + (beta * k1) ** 2 * var_a - 2 * beta * k1 * k2 * cov
    last = k2 * last2 - beta * k1 * last1
    with np.errstate(invalid='ignore', divide='ignore'):
        sd = np.sqrt(np.maximum(var, 0))
        z = (last - mean) / sd
    bad = (na > 0) | (nb > 0) | ~(sd > 1e-12 * (np.abs(mean) + 1)) | ~np.isfinite(z)
    z[bad] = np.nan
    return z


def simulate(c1, c2, o1, o2, r1, r2, beta, start_i, end_i, window, fee, p=0.5, q=0.5):
    state = 'empty'
    w1 = w2 = 0.0
    equity, nav, trades = 1.0, [], 0
    start_i = max(start_i, window)
    if end_i <= start_i:
        return np.array(nav), trades
    zs = rolling_z(c1, c2, r1, r2, beta, start_i, end_i, window)
    with np.errstate(invalid='ignore', divide='ignore'):
        g1s = o1[start_i:end_i] / o1[start_i - 1:end_i - 1] - 1
        g2s = o2[start_i:end_i] / o2[start_i - 1:end_i - 1] - 1
    g1s = np.where(np.isfinite(g1s), g1s, 0.0).tolist()
    g2s = np.where(np.isfinite(g2s), g2s, 0.0).tolist()
    zs = zs.tolist()
    for i in range(end_i - start_i):
        if i > 0:
            g1, g2 = g1s[i], g2s[i]
            growth = 1 + w1 * g1 + w2 * g2
            if growth > 0:
                w1, w2 = w1 * (1 + g1) / growth, w2 * (1 + g2) / growth
            equity *= growth
        z = zs[i]
        target = None
        if z == z:  # 非 NaN
            if z > 1:
                target, state = (1.0, 0.0), 'buy1'
            elif z < -1:
                target, state = (0.0, 1.0), 'buy2'
            elif (state == 'buy1' and z < 0) or (state == 'buy2' and z >= 0):
                target, state = (p, q), 'even'
        if target is not None:
            turnover = abs(target[0] - w1) + abs(target[1] - w2)
            if turnover > 1e-6:
                equity *= 1 - turnover * fee
                trades += 1
            w1, w2 = target
        nav.append(equity)
    return np.array(nav), trades


def perf(nav):
    if len(nav) < 2:
        return np.nan, np.nan, np.nan
    total = nav[-1] / nav[0] - 1
    annual = (nav[-1] / nav[0]) ** (244.0 / len(nav)) - 1
    mdd = (nav / np.maximum.accumulate(nav) - 1).min()
    return total, annual, mdd


def bench_annual(o1, o2, start_i, end_i, window):
    """同期两只股票各 50% 买入持有不动的年化收益，作为对比基准"""
    s = max(start_i, window)
    a, b = o1[s:end_i], o2[s:end_i]
    # 期初有股票还没上市/没有价格时，从两只都有价格的第一天开始买入，之前视为持币
    ok = np.where((a > 0) & (b > 0))[0]
    if len(a) < 2 or len(ok) == 0:
        return np.nan
    j = ok[0]
    nav = np.ones(len(a))
    nav[j:] = 0.5 * a[j:] / a[j] + 0.5 * b[j:] / b[j]
    return perf(nav)[1]


def stability(x, y, parts):
    """把样本等分成 parts 段，各段单独做协整检验，返回 p<0.1 的段数。
    跨行业组合容易是某一段行情造成的巧合，分段都协整才说明关系稳定"""
    n = len(x) // parts
    cnt = 0
    for k in range(parts):
        xs, ys = x[k * n:(k + 1) * n], y[k * n:(k + 1) * n]
        try:
            if len(xs) > 60 and coint_test(ys, xs)[0] < 0.1:
                cnt += 1
        except Exception:
            pass
    return cnt


# ============================================================
# 数据读取
# ============================================================
def load_data(ContextInfo, codes):
    warm = (pd.Timestamp(g.start) - pd.Timedelta(days=int(g.window * 1.6))).strftime('%Y%m%d')
    end = g.oos_end or pd.Timestamp.today().strftime('%Y%m%d')
    if g.download:
        print('正在下载 %d 只股票日线 ...' % len(codes))
        for c in codes:
            try:
                download_history_data(c, '1d', warm, end)
            except Exception as e:
                print('下载 %s 失败: %s' % (c, e))

    def wide(fields, dividend_type):
        res = ContextInfo.get_market_data_ex(fields, codes, period='1d', start_time=warm,
                                             end_time=end, count=-1,
                                             dividend_type=dividend_type,
                                             fill_data=False, subscribe=False)
        out = {}
        for f in fields:
            cols = {}
            for c in codes:
                d = res.get(c)
                if d is not None and len(d) > 0:
                    s = d[f].copy()
                    s.index = pd.to_datetime([str(i)[:8] for i in s.index])
                    cols[c] = s
            out[f] = pd.DataFrame(cols).sort_index()
        return out

    adj = wide(['open', 'close'], 'front_ratio')
    raw = wide(['close', 'volume', 'amount'], 'none')
    idx = adj['close'].index
    st = {}
    for c in codes:
        try:
            name = ContextInfo.get_instrumentdetail(c).get('InstrumentName', '')
        except Exception:
            name = ''
        st[c] = 'ST' in str(name).upper()
    return {
        'close': adj['close'],
        'open': adj['open'].reindex(index=idx, columns=adj['close'].columns),
        'raw': raw['close'].reindex(index=idx, columns=adj['close'].columns),
        'amount': raw['amount'].reindex(index=idx, columns=adj['close'].columns),
        'susp': (raw['volume'].reindex(index=idx, columns=adj['close'].columns).fillna(0) <= 0),
        'st': st,
    }


# ============================================================
def run(ContextInfo):
    codes = list(g.codes) or ContextInfo.get_stock_list_in_sector(g.sector)
    if not codes:
        print('股票池为空：请检查板块名称 %s 或填写 g.codes' % g.sector)
        return
    t0 = time.time()
    print('股票池 %d 只，读取数据 ...' % len(codes))
    data = load_data(ContextInfo, codes)
    print('读取数据用时 %.1f 秒' % (time.time() - t0))
    close = data['close']
    if close.empty:
        print('没有读到行情，请先在「数据管理」下载日线和除权数据，或把 g.download 设为 True')
        return

    ins = (close.index >= pd.Timestamp(g.start)) & (close.index <= pd.Timestamp(g.end))
    if ins.sum() < g.window + 20:
        print('样本内交易日只有 %d 天，至少需要 %d 天，请检查数据或日期' % (ins.sum(), g.window + 20))
        return

    # ---------- 过滤 ----------
    keep = []
    for c in close.columns:
        miss = max(close.loc[ins, c].isna().mean(), data['susp'].loc[ins, c].mean())
        if miss > g.max_missing or data['st'].get(c):
            continue
        if data['amount'].loc[ins, c].mean() < g.min_amount:
            continue
        keep.append(c)
    print('过滤（ST/停牌缺失/成交额）后剩 %d 只' % len(keep))
    if len(keep) < 2:
        return

    # ---------- 相关性初筛 ----------
    logp = np.log(close.loc[ins, keep].ffill())
    corr = logp.corr()
    cands = [(a, b) for a, b in itertools.combinations(keep, 2) if corr.at[a, b] >= g.min_corr]
    print('相关系数 >= %.2f 的组合 %d 个，开始协整检验 ...' % (g.min_corr, len(cands)))

    idx = close.index
    ins_i = np.where(ins)[0]
    oos_i = None
    if g.oos_start:
        m = idx >= pd.Timestamp(g.oos_start)
        if g.oos_end:
            m &= idx <= pd.Timestamp(g.oos_end)
        if m.any():
            oos_i = np.where(m)[0]

    # 预先把数据转成 numpy 数组，循环里不再用 pandas 切片（很慢）
    col = dict((c, i) for i, c in enumerate(keep))
    full_c = close[keep].ffill().values.astype(float)
    full_o = data['open'][keep].ffill().values.astype(float)
    full_r = data['raw'][keep].ffill().values.astype(float)
    ins_c = close.loc[ins, keep].ffill().values.astype(float)
    ins_r = data['raw'].loc[ins, keep].ffill().values.astype(float)
    oos_c = close[keep].iloc[oos_i].ffill().values.astype(float) if oos_i is not None else None
    t_pairs = time.time()

    rows = []
    for n, (a, b) in enumerate(cands, 1):
        if n % 500 == 0:
            el = time.time() - t_pairs
            print('  %d / %d，已用 %.0f 秒，预计还需 %.0f 秒'
                  % (n, len(cands), el, el / n * (len(cands) - n)))
        ia, ib = col[a], col[b]
        ca, cb, ra, rb = ins_c[:, ia], ins_c[:, ib], ins_r[:, ia], ins_r[:, ib]
        m = np.isfinite(ca) & np.isfinite(cb) & np.isfinite(ra) & np.isfinite(rb)
        if m.sum() < g.window + 20:
            continue
        pa = dynamic_scale(ca[m], ra[m])
        pb = dynamic_scale(cb[m], rb[m])

        best = None
        for s1, s2, x, y in ((a, b, pa, pb), (b, a, pb, pa)):
            try:
                res = coint_test(y, x)
            except Exception:
                continue
            if best is None or res[0] < best[0][0]:
                best = (res, s1, s2, x, y)
        if best is None:
            continue
        (pval, alpha, beta, resid), s1, s2, x, y = best
        if pval > g.max_pvalue or beta <= 0:
            continue
        hl = half_life(resid)
        if not (g.min_half_life <= hl <= g.max_half_life):
            continue
        stable = stability(x, y, g.stable_parts)
        if stable < g.min_stable:
            continue

        i1, i2 = col[s1], col[s2]
        c1, c2 = full_c[:, i1], full_c[:, i2]
        o1, o2 = full_o[:, i1], full_o[:, i2]
        r1, r2 = full_r[:, i1], full_r[:, i2]
        nav, trades = simulate(c1, c2, o1, o2, r1, r2, beta,
                               ins_i[0], ins_i[-1] + 1, g.window, g.fee)
        tot, ann, mdd = perf(nav)
        bench = bench_annual(o1, o2, ins_i[0], ins_i[-1] + 1, g.window)
        row = {'security1': s1, 'security2': s2, 'pvalue': pval, 'beta': beta, 'alpha': alpha,
               'half_life': hl, 'corr': corr.at[a, b], 'stable_parts': stable,
               'spread_sigma_pct': resid.std() / y.mean(),
               'ins_return': tot, 'ins_annual': ann, 'ins_maxdd': mdd, 'ins_trades': trades,
               # 超额 = 策略年化 - 两只股票各半持有不动的年化，衡量配对本身贡献
               'ins_bench_annual': bench, 'ins_excess': ann - bench}
        if oos_i is not None:
            nav, trades = simulate(c1, c2, o1, o2, r1, r2, beta,
                                   oos_i[0], oos_i[-1] + 1, g.window, g.fee)
            tot, ann, mdd = perf(nav)
            ox, oy = oos_c[:, i1], oos_c[:, i2]
            om = np.isfinite(ox) & np.isfinite(oy)
            oos_p = coint_test(oy[om], ox[om])[0] if om.sum() > 30 else np.nan
            bench = bench_annual(o1, o2, oos_i[0], oos_i[-1] + 1, g.window)
            row.update({'oos_pvalue': oos_p, 'oos_return': tot, 'oos_annual': ann,
                        'oos_maxdd': mdd, 'oos_trades': trades,
                        'oos_bench_annual': bench, 'oos_excess': ann - bench})
        rows.append(row)
    print('协整检验和回测用时 %.1f 秒，总用时 %.1f 秒' % (time.time() - t_pairs, time.time() - t0))

    if not rows:
        print('没有满足条件的组合，可放宽 g.min_corr / g.max_pvalue / g.max_half_life')
        return

    res = pd.DataFrame(rows).sort_values(['stable_parts', 'pvalue'],
                                         ascending=[False, True]).reset_index(drop=True)
    try:
        res.to_csv(g.out_file, index=False, encoding='gbk')
        print('共 %d 个组合，已保存到 %s' % (len(res), g.out_file))
    except Exception as e:
        print('保存文件失败（%s），只在日志中输出' % e)

    for i, r in res.head(g.top).iterrows():
        line = ('%2d. %s / %s  p=%.4f 稳定%d/%d beta=%.4f 半衰期=%.1f天 相关=%.2f 价差1σ=%.1f%% | '
                '样本内 年化%.1f%% 超额%.1f%% 回撤%.1f%% %d次'
                % (i + 1, r['security1'], r['security2'], r['pvalue'], r['stable_parts'],
                   g.stable_parts, r['beta'], r['half_life'], r['corr'],
                   r['spread_sigma_pct'] * 100, r['ins_annual'] * 100, r['ins_excess'] * 100,
                   r['ins_maxdd'] * 100, r['ins_trades']))
        if 'oos_return' in r:
            line += (' | 样本外 p=%.4f 年化%.1f%% 超额%.1f%% 回撤%.1f%% %d次'
                     % (r['oos_pvalue'], r['oos_annual'] * 100, r['oos_excess'] * 100,
                        r['oos_maxdd'] * 100, r['oos_trades']))
        print(line)

    top = res.iloc[0]
    print('===== 最优组合，填入 qmt_pair_trading.py 的 set_params() =====')
    print("g.security1 = '%s'" % top['security1'])
    print("g.security2 = '%s'" % top['security2'])
    print('g.regression_ratio = %.4f' % top['beta'])
    print('g.test_days = %d' % g.window)
