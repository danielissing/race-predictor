import numpy as np

HEAT_MULT={'cool':1.00,'moderate':1.02,'hot':1.05}
FEEL_MULT={'good':0.98,'ok':1.00,'meh':1.02}
NUM_SIMS = 1000
SIGMA_DAY = 0.035
RHO = 0.4

def fatigue_multiplier(x: float) -> float:
    if x < 0.30: return 1.00
    if x < 0.60: return 1.00 + (x-0.30)*(0.03/0.30)
    return 1.03 + (x-0.60)*(0.05/0.40)

def simulate_etas(legs_meters, speeds_mps, sigmas_rel, leg_ends_x, heat, feel, sims=NUM_SIMS, sigma_day=SIGMA_DAY, rho=RHO):
    n=len(legs_meters); samples=np.zeros((sims,n))
    for s in range(sims):
        day=np.random.normal(0.0, sigma_day); t=0.0
        z=np.random.normal(0.0,1.0,size=speeds_mps.shape[0])
        eps=rho*(day/max(sigma_day,1e-9))+np.sqrt(max(0.0,1-rho**2))*z
        log_sd=np.sqrt(np.log(1+sigmas_rel**2))
        m=np.exp(-0.5*log_sd**2 + log_sd*eps); adj=speeds_mps*m
        for i in range(n):
            base=np.sum(np.where(adj>0, legs_meters[i]/adj, 0.0))
            t+=base*fatigue_multiplier(leg_ends_x[i])*HEAT_MULT.get(heat,1.0)*FEEL_MULT.get(feel,1.0)*(1.0+day)
            samples[s,i]=t
    return np.percentile(samples,10,axis=0), np.percentile(samples,50,axis=0), np.percentile(samples,90,axis=0)
