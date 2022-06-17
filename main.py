import numpy as np
import multiprocessing
from itertools import product


from optimization import algorithm

def accuracy(real_PCT, visiting, result, score = 2):
    # Counting G to G
    GtoG = sum(1 if (PCT < 1478) and ((v == 1) or (s > score)) else 0 for PCT, v , s in zip(real_PCT, visiting, result))
    GtoY = sum(1 if (PCT < 1478) and ((v == 0) and (s < score) and ( s > - score)) else 0 for PCT, v , s in zip(real_PCT, visiting, result))
    GtoR = sum(1 if (PCT < 1478) and ((v == 0) and (s < - score)) else 0 for PCT, v , s in zip(real_PCT, visiting, result))

    RtoR = sum(1 if (PCT > 1478) and ((v == 1) or (s < - score)) else 0 for PCT, v , s in zip(real_PCT, visiting, result))
    RtoY = sum(1 if (PCT > 1478) and ((v == 0) and (s < score) and ( s > - score)) else 0 for PCT, v , s in zip(real_PCT, visiting, result))
    RtoG = sum(1 if (PCT > 1478) and ((v == 0) and (s > score)) else 0 for PCT, v , s in zip(real_PCT, visiting, result))

    visited_branch = sum(visiting)
    return np.array([visited_branch, GtoG, GtoY, GtoR, RtoR, RtoY, RtoG])

def run(score, trial, result_q):
    real_PCT, visiting, result = algorithm(score=score, trial=trial)
    acc = accuracy(real_PCT, visiting, result, score=score)
    result_q.put(np.hstack((np.array([score, trial]), acc)))

def logger(result_q):
    save = np.empty((0, 9))
    while True:
        result = result_q.get()
        if result[0] == 0:
            print('End')
            break
        save = np.vstack((save, result))
        np.savetxt('./result.csv', save, delimiter=',',
                   header='score,trial,visited_branch,GtoG,GtoY,GtoR,RtoR,RtoY,RtoG')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    manager = multiprocessing.Manager()
    result_q = manager.Queue()

    pool = multiprocessing.Pool(processes=8)
    logging = pool.apply_async(logger, (result_q,))

    jobs = []
    for score in np.arange(0.25, 5.25, 0.25):
        for trial in np.arange(0, 10, 1):
            job  = pool.apply_async(run, (score, trial, result_q))
            jobs.append(job)

    for job in jobs:
        job.get()

    result_q.put(np.zeros((9)))

    pool.close()
    pool.join()