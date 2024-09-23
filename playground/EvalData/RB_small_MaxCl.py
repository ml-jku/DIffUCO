if(__name__ == "__main__"):
    import numpy as np
    results = [18.645, 18.824, 18.234]
    print("mean", np.mean(results), np.std(results)/np.sqrt(len(results)))