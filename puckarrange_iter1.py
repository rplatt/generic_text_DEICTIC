# Here, I create a sequence of problems that must be solved successively. The value
# functions learned in the last problem are copied into the next problem. This
# is a form of curriculum learning.
#
import os as os
import numpy as np

# Returns True if average performance of last 50 episodes is at least 5.0
def VerifyProgress(filename):
    episode_rewards = np.loadtxt(filename)
    return(np.mean(episode_rewards[-50:-1]) > 7)
    

def main():

    for i in range(3,10):

#        # parameters: stridey, stridex, filein, fileout, numIter, visualize, obj, numOrientations, useHierarchy

        while True:
            os.system("python puckarrange18.py 28 28 None ./disk_28_2 5000 0 Disks 2 0 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_A_" + str(i) + ".dat")
        
        while True:
            os.system("python puckarrange18.py 28 28 ./disk_28_2 ./disk_28_8 5000 0 Disks 8 0 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_B_" + str(i) + ".dat")
        
        while True:
            os.system("python puckarrange18.py 28 28 ./disk_28_8 ./rect_28_2 5000 0 Blocks 2 0 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_C_" + str(i) + ".dat")
        
        while True:
            os.system("python puckarrange18.py 28 28 ./rect_28_2 ./rect_28_4 5000 0 Blocks 4 0 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_D_" + str(i) + ".dat")
        
        while True:
            os.system("python puckarrange18.py 14 14 ./rect_28_4 ./rect_14_8 8000 0 Blocks 8 0 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_E_" + str(i) + ".dat")
        
        while True:
            os.system("python puckarrange18.py 7 7 ./rect_14_8 ./rect_7_8 8000 0 Blocks 8 0 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_F_" + str(i) + ".dat")
        
        while True:
            os.system("python puckarrange18.py 4 4 ./rect_7_8 ./rect_4_8 8000 0 Blocks 8 1 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_G_" + str(i) + ".dat")
        
        while True:
            os.system("python puckarrange18.py 4 4 ./rect_4_8 ./rect_4_16 8000 0 Blocks 16 1 1")
            if VerifyProgress('./PA18_deictic_rewards.dat'):
                break
        os.system("mv PA18_deictic_rewards.dat PA18_deictic_rewards_H_" + str(i) + ".dat")

#    # parameters: stridey, stridex, filein, fileout, numIter, visualize, obj, numOrientations, useHierarchy
#    os.system("python puckarrange16.py 28 28 None ./disk_28_2 5000 0 Disks 2 0")
#    os.system("python puckarrange16.py 28 28 ./disk_28_2 ./disk_28_8 5000 0 Disks 8 0")
#    os.system("python puckarrange16.py 28 28 ./disk_28_8 ./rect_28_2 5000 0 Blocks 2 0")
#    os.system("python puckarrange16.py 28 28 ./rect_28_2 ./rect_28_4 5000 0 Blocks 4 0")
#    os.system("python puckarrange16.py 14 14 ./rect_28_4 ./rect_14_8 7000 0 Blocks 8 0")
#    os.system("python puckarrange16.py 7 7 ./rect_14_8 ./rect_7_8 5000 0 Blocks 8 0")
#    os.system("python puckarrange16.py 4 4 ./rect_7_8 ./rect_4_8 5000 0 Blocks 8 1")
#    os.system("python puckarrange16.py 4 4 ./rect_4_8 ./rect_4_16 7000 0 Blocks 16 1")
    
#    os.system("python puckarrange15.py 7 7 ./rect_28_4 ./rect_7_8 7000 0 Blocks 8")
    


#    os.system("python puckarrange2.py 14 14 ./whatilearned2020_28 ./whatilearned2020_14 1500")
#    os.system("python puckarrange2.py 7 7 ./whatilearned2020_14 ./whatilearned2020_7 1000")
#    os.system("python puckarrange2.py 4 4 ./whatilearned2020_7 ./whatilearned2020_4 750")
#    os.system("python puckarrange2.py 2 2 ./whatilearned2020_4 ./whatilearned2020_2 500")
#    os.system("python puckarrange2.py 1 2 ./whatilearned2020_2 ./whatilearned2020_1_2 500")
#    os.system("python puckarrange2.py 1 1 ./whatilearned2020_1_2 ./whatilearned2020_1_1 500")


    
if __name__ == '__main__':
    main()
