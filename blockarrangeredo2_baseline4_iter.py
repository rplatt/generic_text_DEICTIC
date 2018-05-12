#
#
#
import os as os

def main():

# BASELINE
#    os.system("python blockarrangeredo2_baseline4.py 2500")
#    os.system("python blockarrangeredo2_baseline4.py 5000")
#    os.system("python blockarrangeredo2_baseline4.py 10000")
#    os.system("python blockarrangeredo2_baseline4.py 20000")
#    os.system("python blockarrangeredo2_baseline4.py 40000")
#    os.system("python blockarrangeredo2_baseline4.py 60000")
#    os.system("python blockarrangeredo2_baseline4.py 100000")

# DEICTIC
#    os.system("python blockarrangeredo2.py 800")
#    os.system("python blockarrangeredo2.py 1000")
#    os.system("python blockarrangeredo2.py 1200")
#    os.system("python blockarrangeredo2.py 2500")
#    os.system("python blockarrangeredo2.py 5000")
#    os.system("python blockarrangeredo2.py 10000")

    for i in range(10):
#        os.system("python blockarrangeredo2.py 5000") # deictic
#        os.system("python blockarrangeredo2_baseline4.py 2500") # baseline 9
#        os.system("python blockarrangeredo2_baseline4.py 10000") # baseline 16
        os.system("python blockarrangeredo2_baseline4.py 20000") # baseline 25
#        os.system("mv BAR2_rewards_9_2500.dat BAR2_rewards_9_2500_" + str(i) + ".dat")
#        os.system("mv BAR2_rewards_16_10000.dat BAR2_rewards_16_10000_" + str(i) + ".dat")
        os.system("mv BAR2_rewards_25_20000.dat BAR2_rewards_25_20000_" + str(i) + ".dat")
    
if __name__ == '__main__':
    main()
