#
#import numbersarrange4 as na
import os as os

def main():

    os.system("python numbersarrange4.py 28 None ./whatilearned28 2000")
    os.system("python numbersarrange4.py 14 ./whatilearned28 ./whatilearned14 2000")
    os.system("python numbersarrange4.py 7 ./whatilearned14 ./whatilearned7 2000")
    os.system("python numbersarrange4.py 4 ./whatilearned7 ./whatilearned4 2000")

#    na.numbersarrange4(28,None,"./whatilearned28",100,reuseModels=None)
#    na.numbersarrange4(14,"./whatilearned28","./whatilearned14",2000,reuseModels=True)
#    na.numbersarrange4(7,"./whatilearned14","./whatilearned7",2000,reuseModels=True)
#    na.numbersarrange4(4,"./whatilearned7","./whatilearned4",2000,reuseModels=True)
    
if __name__ == '__main__':
    main()
