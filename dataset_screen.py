import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    args = parser.parse_args()

    data = np.genfromtxt(f'datasets/{args.dataset}', delimiter=' ', dtype=np.int32)
    if data.shape[1] <= 21 and data.shape[0] >= 700:
        print(args.dataset)
        print(data.shape)