"""Classical baselines on the full numeric feature set, same pool, split seeds and record
identities as the six-component quantum pipeline. Only the encoder differs: no MI
selection, no PCA, no [0,pi] rescaling; a StandardScaler fit on the training partition.
Answers Reviewer 2: does restricting classical models to the six PCA components understate
what a classical detector achieves on these datasets?"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/chibuike/task-aware-qcc/src")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef, accuracy_score
from taqcc.data import DataProcessor, _detect_delimiter, load_split
import os
DD=os.environ.get("TAQCC_DATA_DIR", "data")
DS={"IoTID20":"IoT_Original_Distribution.csv","UNSW-NB15":"UNSW_NB15.csv","Bot-IoT":"UNSW_2018_IoT_Botnet_Final_10_Best.csv"}
def models(seed): return {
  "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed, class_weight="balanced"),
  "SVM_RBF": SVC(kernel="rbf", random_state=seed, class_weight="balanced"),
  "LogisticRegression": LogisticRegression(max_iter=2000, random_state=seed, class_weight="balanced")}
out={"config":{"train_size":200,"test_size":400,"pool_size":5000,"seeds":[0,1,2,3,4],
     "note":"full numeric feature set after the same leakage/categorical drops as taqcc.data; StandardScaler fit on train only; identical pool and stratified split to load_split"},"datasets":{}}
for name,fn in DS.items():
    path=f"{DD}/{fn}"; df=pd.read_csv(path, sep=_detect_delimiter(path), low_memory=False); df.columns=[str(c).strip() for c in df.columns]
    res={}
    for seed in [0,1,2,3,4]:
        proc=DataProcessor(num_qubits=6, random_seed=seed); X,y=proc.prepare_data(df, sample_size=5000)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=400,train_size=200,random_state=seed,stratify=y)
        # sanity: same labels as the quantum split for this seed
        _,_,ytr_q,yte_q=load_split(path,6,200,400,seed=seed); assert (ytr==ytr_q).all() and (yte==yte_q).all()
        sc=StandardScaler().fit(Xtr); Xtr_s,Xte_s=sc.transform(Xtr),sc.transform(Xte)
        for m,clf in models(seed).items():
            clf.fit(Xtr_s,ytr); p=clf.predict(Xte_s)
            res.setdefault(m,{"mcc_seeds":[],"acc_seeds":[]}); res[m]["mcc_seeds"].append(float(matthews_corrcoef(yte,p))); res[m]["acc_seeds"].append(float(accuracy_score(yte,p)))
    for m in res: res[m]["mcc"]=[float(np.mean(res[m]["mcc_seeds"])),float(np.std(res[m]["mcc_seeds"]))]; res[m]["n_features"]=int(X.shape[1])
    out["datasets"][fn]=res
    print(f"{name:<10} ({X.shape[1]} features)  "+"  ".join(f"{m} {res[m]['mcc'][0]:.3f}+-{res[m]['mcc'][1]:.3f}" for m in res))
json.dump(out,open("results/classical_baseline_fullfeat_200.json","w"),indent=1); print("[written] results/classical_baseline_fullfeat_200.json")
