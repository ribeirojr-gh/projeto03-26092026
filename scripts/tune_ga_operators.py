"""Reproduce the GA operator tuning table of docs/development_log.md (stage 6)."""
import numpy as np, itertools
from ga_reaxff.ga import GAConfig, GeneticAlgorithm
OPT=np.array([0.2,0.8,0.35,0.6,0.5])
def rastrigin(x):
    z=5.12*(x-OPT)/2; return float(10*len(z)+np.sum(z**2-10*np.cos(2*np.pi*z)))
for k,etam,pm in itertools.product([2,3],[5,10,20],[None,0.4]):
    ok=[];fin=[]
    for s in range(12):
        r=GeneticAlgorithm(GAConfig(pop_size=60,n_generations=120,seed=100+s,tournament_k=k,eta_m=etam,p_mutation=pm)).run(rastrigin,5)
        ok.append(r.best_loss<0.1); fin.append(r.best_loss)
    print(f"k={k} eta_m={etam} pm={pm}: success {sum(ok)}/12  median best {np.median(fin):.3f}")
