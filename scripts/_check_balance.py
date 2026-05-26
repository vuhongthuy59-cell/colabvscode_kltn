# -*- coding: utf-8 -*-
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

snap = pd.read_csv('data/processed/snapshot_index.csv')
m = pd.read_csv('data/processed/news_mentions.csv')
meta = pd.read_csv('data/processed/ticker_metadata.csv')

tc = m['ticker'].value_counts()
snap_ids = set(snap['article_id'])
snap_m = m[m['article_id'].isin(snap_ids)]
stc = snap_m['ticker'].value_counts()

all_tickers = sorted(set(m['ticker']))
rows = []
for t in all_tickers:
    ind = meta[meta['ticker']==t]['industry'].values[0] if len(meta[meta['ticker']==t])>0 else '?'
    rows.append({'ticker':t,'mentions':int(tc.get(t,0)),'snapshots':int(stc.get(t,0)),'industry':ind})
df = pd.DataFrame(rows)

print('=== TOP 10 NHIEU MENTIONS ===')
for _, r in df.sort_values('mentions',ascending=False).head(10).iterrows():
    print('  %s: %d mentions, %d snapshots (%s)' % (r['ticker'],r['mentions'],r['snapshots'],r['industry']))

print()
print('=== TOP 10 IT MENTIONS ===')
for _, r in df.sort_values('mentions').head(10).iterrows():
    print('  %s: %d mentions, %d snapshots (%s)' % (r['ticker'],r['mentions'],r['snapshots'],r['industry']))

print()
print('=== TOP 10 IT SNAPSHOTS ===')
for _, r in df.sort_values('snapshots').head(10).iterrows():
    print('  %s: %d mentions, %d snapshots (%s)' % (r['ticker'],r['mentions'],r['snapshots'],r['industry']))

print()
low = df[df['snapshots'] < 50]
print('=== TICKERS VOI <50 SNAPSHOTS (%d tickers) ===' % len(low))
for _, r in low.sort_values('snapshots').iterrows():
    print('  %s: %d snapshots, %d mentions (%s)' % (r['ticker'],r['snapshots'],r['mentions'],r['industry']))

vl = df[df['snapshots'] < 20]
print()
print('=== TICKERS VOI <20 SNAPSHOTS (%d tickers) ===' % len(vl))
for _, r in vl.sort_values('snapshots').iterrows():
    print('  %s: %d snapshots, %d mentions (%s)' % (r['ticker'],r['snapshots'],r['mentions'],r['industry']))

print()
print('Stats: mentions mean=%.1f median=%.1f min=%d max=%d' % (df['mentions'].mean(),df['mentions'].median(),df['mentions'].min(),df['mentions'].max()))
print('Stats: snapshots mean=%.1f median=%.1f min=%d max=%d' % (df['snapshots'].mean(),df['snapshots'].median(),df['snapshots'].min(),df['snapshots'].max()))

