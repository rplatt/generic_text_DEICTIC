#
#
#
import os as os

def main():

## BASELINE hyperparameter search
#    os.system("python puckarrange2_baseline4.py 28 28 None None 2500")
#    os.system("python puckarrange2_baseline4.py 28 28 None None 5000")
#    os.system("python puckarrange2_baseline4.py 28 28 None None 10000")
#    os.system("python puckarrange2_baseline4.py 28 28 None None 20000")
#    os.system("python puckarrange2_baseline4.py 28 28 None None 40000")
#    os.system("python puckarrange2_baseline4.py 28 28 None None 60000")
#    os.system("python puckarrange2_baseline4.py 28 28 None None 100000")

## DEICTIC hyperparameter search
#    os.system("python puckarrange2.py 28 28 None None 2500")
#    os.system("python puckarrange2.py 28 28 None None 5000")
#    os.system("python puckarrange2.py 28 28 None None 10000")
#    os.system("python puckarrange2.py 28 28 None None 20000")
#    os.system("python puckarrange2.py 28 28 None None 40000")
#    os.system("python puckarrange2.py 28 28 None None 60000")

# Get data to average
    for i in range(10):
#        os.system("python blockarrangeredo2.py 5000") # deictic
#        os.system("python blockarrangeredo2_baseline4.py 2500") # baseline 9
#        os.system("python blockarrangeredo2_baseline4.py 10000") # baseline 16
#        os.system("python puckarrange2_baseline4.py 28 28 None None 5000") # baseline 9
#        os.system("python puckarrange2_baseline4.py 28 28 None None 10000") # baseline 16
#        os.system("python puckarrange2_baseline4.py 28 28 None None 40000") # baseline 25
        os.system("python puckarrange2.py 28 28 None None 10000") # deictic
#        os.system("mv BAR2_rewards_9_2500.dat BAR2_rewards_9_2500_" + str(i) + ".dat")
#        os.system("mv BAR2_rewards_16_10000.dat BAR2_rewards_16_10000_" + str(i) + ".dat")
#        os.system("mv BAR2_rewards_25_20000.dat BAR2_rewards_25_20000_" + str(i) + ".dat")
#        os.system("mv PA2_rewards_9_5000.dat PA2_rewards_9_5000_" + str(i) + ".dat")
#        os.system("mv PA2_rewards_16_10000.dat PA2_rewards_16_10000_" + str(i) + ".dat")
        os.system("mv PA2_deictic_rewards_64_10000.dat PA2_deictic_rewards_64_10000_" + str(i) + ".dat")
    
if __name__ == '__main__':
    main()
