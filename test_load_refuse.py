from dataset import get_pathdata
import test_refuse
import model_repo
import torch
import random
import numpy as np
import os
import argparse


if __name__ == '__main__':

    # # Refused_data
    testdata_path = r'SSL/wval_data/Refused/test_data.npz'

    # Test_data
    testdata_path = r'SSL/wval_data/test_data.npz'


    os.makedirs('Test/', exist_ok=True)
    save_path = r'Test/'  # save predicted results

    # model path
    pth = r'SSL/result/model_best.pth.tar'
    netname = model_repo.MLP

    config = {
        'netname': netname.Net,
        'dataset': {'test': get_pathdata(testdata_path),},
        'pth_repo': pth,
        'test_path': save_path,
    }

    tester = test_refuse.Test(config)
    accuracys = []
    for i in range(1):
        print(f"Running test iteration {i + 1}...")
        accuracy = tester.start()
        accuracy = accuracy*100
        accuracys.append(accuracy)
        print(f"Test iteration {i + 1} completed.")

    accuracys = np.array(accuracys)
    print("\t".join(map(str, accuracys)))
