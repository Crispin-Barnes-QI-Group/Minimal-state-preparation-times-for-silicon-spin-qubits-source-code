import numpy as np

def bootstrap(MET, ts, number_bootstrap_samples, confidence_interval):
    # Bootstrap resampling
    resamples = np.random.choice(MET, (number_bootstrap_samples, len(MET)))

    # Generate the distribution for each resample
    hists = np.apply_along_axis(lambda x: np.histogram(x, ts)[0], 1, resamples)
    hists = hists/len(MET)
    cumulatives = np.cumsum(hists, axis=-1)
    
    # Compute the standard error in the cumulative distribution
    sigmas = np.maximum(1/len(MET), np.sqrt(np.mean(np.square(cumulatives), axis=0)-np.square(np.mean(cumulatives, axis=0))))

    # Compute confidence intervals
    ordered = np.sort(cumulatives, axis=0)
    index = int(number_bootstrap_samples*(100-confidence_interval)/200)
    return ordered[index], ordered[-index-1], sigmas
        