# Here, I create a sequence of problems that must be solved successively. The value
# functions learned in the last problem are copied into the next problem. This
# is a form of curriculum learning.
#
import os as os

def main():

    os.system("python puckarrange2.py 28 28 None ./whatilearned2020_28 8000")
    os.system("python puckarrange2.py 14 14 ./whatilearned2020_28 ./whatilearned2020_14 1500")
    os.system("python puckarrange2.py 7 7 ./whatilearned2020_14 ./whatilearned2020_7 1000")
    os.system("python puckarrange2.py 4 4 ./whatilearned2020_7 ./whatilearned2020_4 750")
    os.system("python puckarrange2.py 2 2 ./whatilearned2020_4 ./whatilearned2020_2 500")
    os.system("python puckarrange2.py 1 2 ./whatilearned2020_2 ./whatilearned2020_1_2 500")
    os.system("python puckarrange2.py 1 1 ./whatilearned2020_1_2 ./whatilearned2020_1_1 500")

#    na.numbersarrange4(28,None,"./whatilearned28",100,reuseModels=None)
#    na.numbersarrange4(14,"./whatilearned28","./whatilearned14",2000,reuseModels=True)
#    na.numbersarrange4(7,"./whatilearned14","./whatilearned7",2000,reuseModels=True)
#    na.numbersarrange4(4,"./whatilearned7","./whatilearned4",2000,reuseModels=True)
    
if __name__ == '__main__':
    main()
