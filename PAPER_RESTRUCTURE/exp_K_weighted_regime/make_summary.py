import pandas as pd, numpy as np
from scipy import stats
r=pd.read_csv('results.csv'); c=pd.read_csv('controls.csv'); sg=pd.read_csv('spectral_gap.csv').set_index('network')
NET=["ca-GrQc","email-Eu-core","wiki-Vote","ca-HepTh","ca-CondMat","email-Enron","com-DBLP","com-Amazon"]
r=r[r.network!='email-Eu-core-labeled']
def cell(df,k): return df[k].mean(), (df[k].std(ddof=1) if len(df)>1 else float('nan'))
L=[]
L.append("| network | 1/alpha | sampler | alpha | retention | weighted dQ_fixed (a) | p(=0) | unweighted dQ_fixed (same topology) | honest transfer vs base_mean (b) | vs base_best | n_clusters (P_w) | n_clusters base |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for net in NET:
    ia=sg.loc[net,'inv_alpha']
    for samp,a in [("paper",0.8),("paper",1.0),("calibrated",0.9)]:
        d=r[(r.network==net)&(r.sampler==samp)&(np.isclose(r.alpha,a))]
        if not len(d): continue
        cu=c[(c.network==net)&(c.sampler==samp)&(np.isclose(c.alpha,a))&(~c.weighted)]
        t=stats.ttest_1samp(d.dQ_fixed_w,0.0)
        um = f"{cu.dQ_fixed.mean():+.4f} ± {cu.dQ_fixed.std(ddof=1):.4f}" if len(cu)>1 else "—"
        L.append(f"| {net} | {ia:.0f} | {samp} | {a} | {d.retention.mean():.3f} | "
                 f"{d.dQ_fixed_w.mean():+.4f} ± {d.dQ_fixed_w.std(ddof=1):.4f} | {t.pvalue:.2f} | {um} | "
                 f"{d.transfer_vs_mean.mean():+.4f} ± {d.transfer_vs_mean.std(ddof=1):.4f} | "
                 f"{d.transfer_vs_best.mean():+.4f} | {d.nc_Pw.mean():.0f} | {d.nc_base_mean.mean():.0f} |")
print("\n".join(L))
print()
print("### retention-matched control (calibrated Bernoulli at alpha = paper-sampler realized retention)")
print("| network | retention | dQ_fixed weighted | dQ_fixed unweighted | transfer weighted | transfer unweighted | nc w | nc unw |")
print("|---|---|---|---|---|---|---|---|")
for net in NET:
    cm=c[(c.network==net)&(c.sampler=='calibrated_matched')]
    w=cm[cm.weighted]; u=cm[~cm.weighted]
    if not len(w): continue
    print(f"| {net} | {w.retention.mean():.3f} | {w.dQ_fixed.mean():+.4f} ± {w.dQ_fixed.std(ddof=1):.4f} | "
          f"{u.dQ_fixed.mean():+.4f} ± {u.dQ_fixed.std(ddof=1):.4f} | "
          f"{w.transfer_vs_mean.mean():+.4f} | {u.transfer_vs_mean.mean():+.4f} | {w.nc_P.mean():.0f} | {u.nc_P.mean():.0f} |")
print()
print("### calibrated alpha=0.9: weighted vs unweighted pipeline (identical topology)")
print("| network | transfer weighted | transfer unweighted | diff (w - unw) | dQ_fixed w | dQ_fixed unw |")
print("|---|---|---|---|---|---|")
for net in NET:
    w=r[(r.network==net)&(r.sampler=='calibrated')]
    u=c[(c.network==net)&(c.sampler=='calibrated')&(~c.weighted)]
    print(f"| {net} | {w.transfer_vs_mean.mean():+.4f} | {u.transfer_vs_mean.mean():+.4f} | "
          f"{w.transfer_vs_mean.mean()-u.transfer_vs_mean.mean():+.4f} | {w.dQ_fixed_w.mean():+.4f} | {u.dQ_fixed.mean():+.4f} |")
print()
print("### runtime")
rt=r.groupby(['network','sampler','alpha']).agg(t_sp=('t_sparsify','mean'),t_ld=('t_leiden_sparse','mean'),t_or=('t_leiden_orig_mean','mean'),ret=('retention','mean'))
rt['pipeline']=rt.t_sp+rt.t_ld; rt['speedup']=rt.t_or/rt['pipeline']
print("| network | sampler | alpha | retention | t_sparsify | t_leiden(weighted,sparse) | t_leiden(orig) | pipeline speedup |")
print("|---|---|---|---|---|---|---|---|")
for (net,s,a),row in rt.iterrows():
    print(f"| {net} | {s} | {a} | {row.ret:.3f} | {row.t_sp:.3f} | {row.t_ld:.3f} | {row.t_or:.3f} | {row.speedup:.2f}x |")
