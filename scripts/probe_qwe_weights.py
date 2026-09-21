"""Why accuracy weighting (QWE3) survives on one dataset and not the others.

For data split 0 at six qubits and p1 = 0.01, report each member's validation accuracy (the
QWE3 weight) and the constant label a collapsed member emits. CPU only.

  python scripts/probe_qwe_weights.py /path/to/csvs
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from taqcc.data import load_split
from taqcc.feature_maps import make_feature_map
from taqcc.kernels import gram_pair
out={}
for ds in ("UNSW_NB15.csv","IoT_Original_Distribution.csv","UNSW_2018_IoT_Botnet_Final_10_Best.csv"):
    Xtr,Xte,ytr,yte=load_split(sys.argv[1].rstrip("/")+"/"+ds,6,200,400,seed=0)
    idx=np.arange(len(ytr)); ifit,ival=train_test_split(idx,test_size=0.2,stratify=ytr,random_state=0)
    r={"train_class1_rate":float(ytr.mean()),"test_class1_rate":float(yte.mean())}
    for lab,m in (("Z",("Z",1,"full")),("ZZ",("ZZ",2,"full")),("Pauli",("Pauli",1,"full"))):
        K,R=gram_pair(make_feature_map(6,*m),Xtr,Xte,p1=0.01,p2=None,gpu=False)
        svc=SVC(kernel="precomputed",class_weight="balanced",random_state=0,C=1.0).fit(K[np.ix_(ifit,ifit)],ytr[ifit])
        pv=svc.predict(K[np.ix_(ival,ifit)])
        full=SVC(kernel="precomputed",class_weight="balanced",random_state=0,C=1.0).fit(K,ytr).predict(R)
        r[lab]={"val_acc":float(accuracy_score(ytr[ival],pv)),"val_pred_class1_rate":float(pv.mean()),"test_pred_class1_rate":float(full.mean())}
    out[ds]=r; print(ds,json.dumps(r),flush=True)
json.dump(out,open(Path(__file__).resolve().parents[1] / "results/corrected/qwe_weight_probe.json","w"),indent=1)
