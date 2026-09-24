# coding: utf-8
"""
配对交易选股脚本 —— 为 qmt_pair_trading.py 挑选股票对并估计参数

流程：
  1. 读取行情（CSV 长表 或 QMT 的 xtquant.xtdata）
  2. 过滤：ST、数据缺失/停牌过多、成交额过低、（可选）不同行业
  3. 两两计算对数价格相关性，保留 >= --min-corr 的组合
  4. Engle-Granger 协整检验（两个方向都试，取 p 值小的方向）
  5. 估计回归系数 beta（P2 = alpha + beta * P1）、价差半衰期、价差波动
  6. 用 qmt_pair_trading.py 相同的规则（250 日 z-score，±1 进场，穿越 0 恢复 p/q）
     模拟回测，计入手续费；可指定样本外区间验证
  7. 输出结果 CSV，并打印可直接填入 QMT 策略的参数

价格口径与 QMT 策略一致：用前复权价，并按每个窗口最后一天的真实收盘价缩放
（即聚宽的「动态前复权」），这样估计出的 beta 可以直接当 g.regression_ratio 用。

依赖：pip install pandas numpy statsmodels

用法示例：
  # 1) CSV（长表，列：date, asset, close，可选 open, raw_close, amount,
  #    industry, is_st, suspendFlag —— 与 market_data_first_100.csv 格式相同）
  python pair_selector.py --source csv --csv market_data.csv \\
      --start 2020-01-01 --end 2023-12-31 --oos-start 2024-01-01 --oos-end 2025-12-31

  # 2) 在装了 QMT / MiniQMT 的电脑上用 xtdata 取数（需先启动 QMT 客户端）
  python pair_selector.py --source xtdata --sector 银行 \\
      --start 2020-01-01 --end 2023-12-31 --oos-start 2024-01-01

  # 只在指定股票里配对
  python pair_selector.py --source xtdata --codes 600036.SH,601166.SH,600000.SH,601398.SH \\
      --start 2020-01-01 --end 2023-12-31
"""
import argparse
import itertools
import math
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint


# ============================================================
# 数据读取：统一返回宽表 dict
#   close  : 前复权收盘价（index=日期，columns=代码）
#   open   : 前复权开盘价（没有则用 close）
#   raw    : 不复权收盘价（没有则为 None，此时不做动态缩放）
#   amount : 成交额（没有则为 None）
#   st     : 是否 ST（0/1，没有则为 None）
#   susp   : 是否停牌（0/1，没有则为 None）
#   industry : Series 代码 -> 行业（没有则为 None）
# ============================================================
def load_csv(path, codes=None):
    df = pd.read_csv(path, encoding='utf-8-sig', dtype={'asset': str})
    df['date'] = pd.to_datetime(df['date'].astype(str))
    if codes:
        df = df[df['asset'].isin(codes)]
    if df.empty:
        sys.exit('CSV 中没有符合条件的数据')

    def wide(col):
        if col not in df.columns:
            return None
        return df.pivot_table(index='date', columns='asset', values=col, aggfunc='last').sort_index()

    data = {
        'close': wide('close'),
        'open': wide('open'),
        'raw': wide('raw_close'),
        'amount': wide('amount'),
        'st': wide('is_st'),
        'susp': wide('suspendFlag'),
        'industry': None,
    }
    if data['open'] is None:
        data['open'] = data['close']
    if 'industry' in df.columns:
        ind = df.sort_values('date').groupby('asset')['industry'].last()
        if not (ind.isin(['UNKNOWN', '', None]) | ind.isna()).all():
            data['industry'] = ind
    return data


def load_xtdata(codes, sector, start, end):
    try:
        from xtquant import xtdata
    except ImportError:
        sys.exit('未找到 xtquant，请在 QMT 自带的 Python 或安装了 xtquant 的环境中运行')

    if not codes:
        if not sector:
            sys.exit('xtdata 模式需要 --codes 或 --sector')
        codes = xtdata.get_stock_list_in_sector(sector)
        if not codes:
            sys.exit('板块 %s 没有成分股，请检查板块名称（可用 xtdata.get_sector_list() 查看）' % sector)
    print('股票数量: %d，正在下载/读取日线数据 ...' % len(codes))

    s, e = start.replace('-', ''), end.replace('-', '')
    for c in codes:
        xtdata.download_history_data(c, '1d', s, e)

    def wide(fields, dividend_type):
        res = xtdata.get_market_data_ex(fields, codes, period='1d', start_time=s, end_time=e,
                                        dividend_type=dividend_type, fill_data=False)
        out = {}
        for f in fields:
            cols = {}
            for c in codes:
                d = res.get(c)
                if d is not None and len(d):
                    cols[c] = d[f]
            w = pd.DataFrame(cols)
            w.index = pd.to_datetime(w.index.astype(str).str[:8])
            out[f] = w.sort_index()
        return out

    adj = wide(['open', 'close'], 'front_ratio')
    raw = wide(['close', 'volume', 'amount'], 'none')
    st, industry = {}, {}
    for c in codes:
        detail = xtdata.get_instrument_detail(c) or {}
        name = detail.get('InstrumentName', '')
        st[c] = 1 if 'ST' in name.upper() else 0
    return {
        'close': adj['close'],
        'open': adj['open'],
        'raw': raw['close'],
        'amount': raw['amount'],
        'st': pd.DataFrame([st], index=[adj['close'].index[-1]]) if len(adj['close']) else None,
        'susp': (raw['volume'] <= 0).astype(int),
        'industry': None,
    }


# ============================================================
# 统计工具
# ============================================================
def dynamic_scale(adj, raw):
    """把前复权序列缩放到最后一天等于真实收盘价（聚宽动态前复权）"""
    if raw is None or not raw[-1] > 0 or not adj[-1] > 0:
        return adj
    return adj * (raw[-1] / adj[-1])


def ols(y, x):
    """y = a + b*x，返回 (a, b, 残差)"""
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef[0], coef[1], y - X.dot(coef)


def half_life(resid):
    """Δε = c + λ·ε(t-1)，半衰期 = -ln2/λ"""
    lag, delta = resid[:-1], np.diff(resid)
    _, lam, _ = ols(delta, lag)
    return -math.log(2) / lam if lam < 0 else np.inf


# ============================================================
# 模拟 qmt_pair_trading.py 的交易规则
# ============================================================
def simulate(close1, close2, open1, open2, raw1, raw2, beta, start_i, end_i,
             window=250, p=0.5, q=0.5, fee=0.001):
    """
    close/open/raw 为对齐后的 numpy 数组。第 t 天用 [t-window, t-1] 的收盘价算 z，
    按第 t 天开盘价调仓。fee 为单边换手费率（佣金+印花税+滑点的平均）。
    返回 (净值序列, 调仓次数)
    """
    state = 'empty'
    w1 = w2 = 0.0              # 当前持仓市值占比（开盘时）
    equity, nav, trades = 1.0, [], 0
    start_i = max(start_i, window)
    for t in range(start_i, end_i):
        if t > start_i:
            # 从上一天开盘到今天开盘的收益
            r1 = open1[t] / open1[t - 1] - 1
            r2 = open2[t] / open2[t - 1] - 1
            r1 = r1 if np.isfinite(r1) else 0.0
            r2 = r2 if np.isfinite(r2) else 0.0
            growth = 1 + w1 * r1 + w2 * r2
            if growth > 0:
                w1, w2 = w1 * (1 + r1) / growth, w2 * (1 + r2) / growth
            equity *= growth

        a = close1[t - window:t]
        b = close2[t - window:t]
        if raw1 is not None:
            a = dynamic_scale(a, raw1[t - window:t])
            b = dynamic_scale(b, raw2[t - window:t])
        spread = b - beta * a
        sd = spread.std()
        target = None
        if sd > 0 and np.isfinite(sd):
            z = (spread[-1] - spread.mean()) / sd
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
            if len(xs) > 60 and coint(ys, xs)[1] < 0.1:
                cnt += 1
        except Exception:
            pass
    return cnt


# ============================================================
def main():
    ap = argparse.ArgumentParser(description='配对交易选股（协整检验 + 规则回测）')
    ap.add_argument('--source', choices=['csv', 'xtdata'], default='csv')
    ap.add_argument('--csv', help='CSV 路径（source=csv）')
    ap.add_argument('--codes', help='逗号分隔的股票代码，如 600036.SH,601166.SH')
    ap.add_argument('--sector', help='xtdata 板块名，如 银行、沪深300')
    ap.add_argument('--start', required=True, help='样本内（选股）开始日期 YYYY-MM-DD')
    ap.add_argument('--end', required=True, help='样本内结束日期')
    ap.add_argument('--oos-start', help='样本外开始日期（可选）')
    ap.add_argument('--oos-end', help='样本外结束日期（默认到数据末尾）')
    ap.add_argument('--window', type=int, default=250, help='z-score 窗口，同 g.test_days')
    ap.add_argument('--min-corr', type=float, default=0.8, help='对数价格最低相关系数')
    ap.add_argument('--max-pvalue', type=float, default=0.05, help='协整检验最大 p 值')
    ap.add_argument('--min-half-life', type=float, default=2, help='最短半衰期（天）')
    ap.add_argument('--max-half-life', type=float, default=60, help='最长半衰期（天）')
    ap.add_argument('--min-amount', type=float, default=5e7, help='样本内日均成交额下限（元）')
    ap.add_argument('--max-missing', type=float, default=0.05, help='允许的缺失/停牌天数比例')
    ap.add_argument('--same-industry', action='store_true', help='只在同行业内配对（需要行业数据）')
    ap.add_argument('--fee', type=float, default=0.001, help='模拟回测单边换手费率')
    ap.add_argument('--stable-parts', type=int, default=3, help='稳定性检验：样本内分成几段')
    ap.add_argument('--min-stable', type=int, default=2, help='至少几段协整（p<0.1）才保留，0 不过滤')
    ap.add_argument('--top', type=int, default=20, help='打印前 N 个组合')
    ap.add_argument('--out', default='pair_candidates.csv', help='结果输出文件')
    args = ap.parse_args()

    codes = [c.strip() for c in args.codes.split(',')] if args.codes else None
    if args.source == 'csv':
        if not args.csv:
            sys.exit('source=csv 时需要 --csv')
        data = load_csv(args.csv, codes)
    else:
        # 多取一个窗口的历史给 z-score 预热
        warm = (pd.Timestamp(args.start) - pd.Timedelta(days=int(args.window * 1.6))).strftime('%Y-%m-%d')
        end = args.oos_end or pd.Timestamp.today().strftime('%Y-%m-%d')
        data = load_xtdata(codes, args.sector, warm, end)

    close = data['close']
    ins = (close.index >= pd.Timestamp(args.start)) & (close.index <= pd.Timestamp(args.end))
    if ins.sum() < args.window + 20:
        sys.exit('样本内交易日只有 %d 天，至少需要 %d 天' % (ins.sum(), args.window + 20))

    # ---------- 过滤 ----------
    keep = []
    reasons = {}
    for c in close.columns:
        s = close.loc[ins, c]
        miss = s.isna().mean()
        if data['susp'] is not None and c in data['susp']:
            miss = max(miss, data['susp'].loc[ins, c].fillna(0).mean())
        if miss > args.max_missing:
            reasons[c] = '缺失/停牌 %.0f%%' % (miss * 100)
            continue
        if data['st'] is not None and c in data['st'] and data['st'][c].fillna(0).max() > 0:
            reasons[c] = 'ST'
            continue
        if data['amount'] is not None and c in data['amount']:
            amt = data['amount'].loc[ins, c].mean()
            if amt < args.min_amount:
                reasons[c] = '日均成交额 %.0f 万' % (amt / 1e4)
                continue
        keep.append(c)
    print('股票 %d 只，过滤后剩 %d 只' % (close.shape[1], len(keep)))
    if len(keep) < 2:
        for c, r in reasons.items():
            print('  剔除 %s: %s' % (c, r))
        sys.exit('可用股票不足 2 只')

    if args.same_industry and data['industry'] is None:
        print('警告：没有行业数据，忽略 --same-industry')
        args.same_industry = False

    # ---------- 相关性初筛 ----------
    logp = np.log(close.loc[ins, keep].ffill())
    corr = logp.corr()
    cands = []
    for a, b in itertools.combinations(keep, 2):
        if args.same_industry and data['industry'].get(a) != data['industry'].get(b):
            continue
        if corr.at[a, b] >= args.min_corr:
            cands.append((a, b))
    print('相关系数 >= %.2f 的组合: %d 个，开始协整检验 ...' % (args.min_corr, len(cands)))

    # ---------- 协整检验 + 参数估计 + 模拟回测 ----------
    idx = close.index
    oos_mask = None
    if args.oos_start:
        oos_mask = idx >= pd.Timestamp(args.oos_start)
        if args.oos_end:
            oos_mask &= idx <= pd.Timestamp(args.oos_end)

    def arr(frame, c):
        return None if frame is None else frame[c].ffill().values.astype(float)

    rows = []
    for n, (a, b) in enumerate(cands, 1):
        if n % 200 == 0:
            print('  %d / %d' % (n, len(cands)))
        pa = close.loc[ins, a].ffill().dropna()
        pb = close.loc[ins, b].ffill().dropna()
        common = pa.index.intersection(pb.index)
        pa, pb = pa[common].values, pb[common].values
        if len(common) < args.window + 20:
            continue
        # 与 QMT 一致：缩放到样本内最后一天的真实价格
        if data['raw'] is not None:
            ra = data['raw'].loc[common, a].ffill().values
            rb = data['raw'].loc[common, b].ffill().values
            pa, pb = dynamic_scale(pa, ra), dynamic_scale(pb, rb)

        best = None
        # 两个方向：security2 = y，security1 = x
        for s1, s2, x, y in ((a, b, pa, pb), (b, a, pb, pa)):
            try:
                _, pval, _ = coint(y, x)
            except Exception:
                continue
            if best is None or pval < best[0]:
                best = (pval, s1, s2, x, y)
        if best is None or best[0] > args.max_pvalue:
            continue
        pval, s1, s2, x, y = best
        alpha, beta, resid = ols(y, x)
        if beta <= 0:
            continue
        hl = half_life(resid)
        if not (args.min_half_life <= hl <= args.max_half_life):
            continue
        stable = stability(x, y, args.stable_parts)
        if stable < args.min_stable:
            continue

        c1, c2 = arr(close, s1), arr(close, s2)
        o1, o2 = arr(data['open'], s1), arr(data['open'], s2)
        r1, r2 = arr(data['raw'], s1), arr(data['raw'], s2)
        ins_i = np.where(ins)[0]
        nav, trades = simulate(c1, c2, o1, o2, r1, r2, beta,
                               ins_i[0], ins_i[-1] + 1, args.window, fee=args.fee)
        tot, ann, mdd = perf(nav)
        bench = bench_annual(o1, o2, ins_i[0], ins_i[-1] + 1, args.window)
        row = {
            'security1': s1, 'security2': s2,
            'pvalue': pval, 'beta': beta, 'alpha': alpha,
            'half_life': hl, 'corr': corr.at[a, b], 'stable_parts': stable,
            # 1 个标准差价差占股票2价格的比例：太小则赚不够手续费
            'spread_sigma_pct': resid.std() / y.mean(),
            'price1': x[-1], 'price2': y[-1],
            'ins_return': tot, 'ins_annual': ann, 'ins_maxdd': mdd, 'ins_trades': trades,
            # 超额 = 策略年化 - 两只股票各半持有不动的年化，衡量配对本身贡献
            'ins_bench_annual': bench, 'ins_excess': ann - bench,
        }
        if oos_mask is not None and oos_mask.any():
            oi = np.where(oos_mask)[0]
            nav, trades = simulate(c1, c2, o1, o2, r1, r2, beta,
                                   oi[0], oi[-1] + 1, args.window, fee=args.fee)
            tot, ann, mdd = perf(nav)
            ob = pd.Series(oos_mask, index=idx)
            ox = close.loc[ob.values, s1].ffill().dropna()
            oy = close.loc[ob.values, s2].ffill().dropna()
            cm = ox.index.intersection(oy.index)
            try:
                oos_p = coint(oy[cm].values, ox[cm].values)[1] if len(cm) > 30 else np.nan
            except Exception:
                oos_p = np.nan
            bench = bench_annual(o1, o2, oi[0], oi[-1] + 1, args.window)
            row.update({'oos_pvalue': oos_p, 'oos_return': tot, 'oos_annual': ann,
                        'oos_maxdd': mdd, 'oos_trades': trades,
                        'oos_bench_annual': bench, 'oos_excess': ann - bench})
        rows.append(row)

    if not rows:
        sys.exit('没有找到满足条件的组合，可放宽 --min-corr / --max-pvalue / --max-half-life')

    res = pd.DataFrame(rows).sort_values(['stable_parts', 'pvalue'],
                                         ascending=[False, True]).reset_index(drop=True)
    res.to_csv(args.out, index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    show = res.head(args.top).copy()
    fmt_pct = [c for c in show.columns if c.endswith(('return', 'annual', 'maxdd', 'sigma_pct', 'excess'))]
    for c in fmt_pct:
        show[c] = show[c].map(lambda v: '' if pd.isna(v) else '%.1f%%' % (v * 100))
    for c in ['pvalue', 'oos_pvalue']:
        if c in show:
            show[c] = show[c].map(lambda v: '' if pd.isna(v) else '%.4f' % v)
    for c in ['beta', 'alpha', 'half_life', 'corr', 'price1', 'price2']:
        show[c] = show[c].map(lambda v: '%.3f' % v)
    print('\n共 %d 个组合满足条件，结果已保存到 %s\n' % (len(res), args.out))
    print(show.to_string())

    top = res.iloc[0]
    print('\n# ===== 最优组合，填入 qmt_pair_trading.py 的 set_params() =====')
    print("g.security1 = '%s'" % top['security1'])
    print("g.security2 = '%s'" % top['security2'])
    print('g.regression_ratio = %.4f' % top['beta'])
    print('g.test_days = %d' % args.window)
    print('# 协整 p=%.4f，半衰期 %.1f 天，样本内规则回测收益 %.1f%%'
          % (top['pvalue'], top['half_life'], top['ins_return'] * 100))
    if 'oos_return' in top:
        print('# 样本外 p=%.4f，规则回测收益 %.1f%%（样本外结果更可信）'
              % (top['oos_pvalue'], top['oos_return'] * 100))


if __name__ == '__main__':
    main()
