#!/bin/bash
# For LTCC dataset
# python main.py --dataset ltcc --gpu 2,3 -p 12905
# python main_test_ltcc.py --dataset ltcc --gpu 0 -p 12905
python main_test_vcc.py --dataset vcclothes --gpu 0 -p 12905
# For PRCC dataset
# python main.py --dataset prcc --gpu 0 -p 12905
# For VC-Clothes dataset.
# python main.py --dataset vcclothes --gpu 0,1 -p 12901
# python main.py --dataset vcclothes_cc --gpu 0,1
# python main.py --dataset vcclothes_sc --gpu 0,1
# For DeepChange dataset.
#python main.py --dataset deepchange --gpu 0,1
