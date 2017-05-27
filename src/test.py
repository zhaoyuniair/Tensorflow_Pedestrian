import  numpy as np

def test1():
    # test for string
    sStr1 = 'strchr'
    sStr2 = 'strch'
    print 's1=', sStr1
    print sStr1.find('rch')
    print sStr1.find('rcm')
    print cmp(sStr1, 'rch')
    print cmp(sStr1, 'rcr')

if __name__ == '__main__':
    test1()


