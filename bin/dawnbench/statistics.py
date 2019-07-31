import sys
import numpy as np

if __name__ == '__main__':
    filenames = sys.argv[1:]
    scores = []
    for filename in filenames:
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) < 2:
                print('WARNING!', filename, 'is invalid')
                continue
            score = float(lines[-1].split('\t')[-1])
            scores.append(score)

    print('num:', len(scores))
    print('count:', np.sum(np.array(scores) >= 94.0))
    print('average:', np.average(scores))
    print('median:', np.median(scores))
    print('min:', np.min(scores))
    print('max:', np.max(scores))
