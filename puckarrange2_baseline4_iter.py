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
#        os.system("python blockarrangeredo2_baseline4.py 15000") # baseline 16
#        os.system("python puckarrange2_baseline4.py 28 28 None None 5000") # baseline 9
#        os.system("python puckarrange2_baseline4.py 28 28 None None 10000") # baseline 16
#        os.system("python puckarrange2_baseline4.py 28 28 None None 15000") # baseline 16
#        os.system("python puckarrange2_baseline4.py 28 28 None None 60000") # baseline 25
#        os.system("python puckarrange2.py 28 28 None None 10000") # deictic
#        os.system("python puckarrange2.py 14 14 None None 20000") # deictic
#        os.system("mv BAR2_rewards_9_2500.dat BAR2_rewards_9_2500_" + str(i) + ".dat")
#        os.system("mv BAR2_rewards_16_10000.dat BAR2_rewards_16_10000_" + str(i) + ".dat")
#        os.system("mv BAR2_rewards_25_20000.dat BAR2_rewards_25_20000_" + str(i) + ".dat")
#        os.system("mv PA2_rewards_9_5000.dat PA2_rewards_9_5000_" + str(i) + ".dat")
#        os.system("mv PA2_rewards_16_10000.dat PA2_rewards_16_10000_" + str(i) + ".dat")
#        os.system("mv PA2_rewards_16_15000.dat PA2_rewards_16_15000_" + str(i) + ".dat")
#        os.system("mv PA2_rewards_25_60000.dat PA2_rewards_25_60000_" + str(i) + ".dat")
#        os.system("mv PA2_deictic_rewards_16_15000.dat PA2_deictic_rewards_16_15000_" + str(i) + ".dat")
#        os.system("mv PA2_deictic_rewards_25_10000.dat PA2_deictic_rewards_25_10000_" + str(i) + ".dat")
#        os.system("mv PA2_deictic_rewards_25_20000.dat PA2_deictic_rewards_25_20000_14_" + str(i) + ".dat")
#        os.system("mv PA2_deictic_rewards_64_10000.dat PA2_deictic_rewards_64_10000_" + str(i) + ".dat")
        
        # do curriculum learning
        os.system("python puckarrange2.py 28 28 None ./PA_28 2500")
        os.system("mv PA2_deictic_rewards_25_2500.dat PA2_deictic_rewards_25_2500_28_" + str(i) + ".dat")
        
        os.system("python puckarrange2.py 14 14 ./PA_28 ./PA_14 2500")
        os.system("mv PA2_deictic_rewards_25_2500.dat PA2_deictic_rewards_25_2500_14_" + str(i) + ".dat")

    
if __name__ == '__main__':
    main()
