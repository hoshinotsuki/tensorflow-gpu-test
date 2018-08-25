from __future__ import print_function

import numpy as np

a=[1,2,3,4,5]
b=(1,2,3,4,5)
c=np.arange(5)
d="zhang"
zz=zip(a,b,c,d)
print(zz)#<zip object at 0x0000019145EDD188>只输出object
print(list(zz))#[(1, 1, 0, 'z'), (2, 2, 1, 'h'), (3, 3, 2, 'a'), (4, 4, 3, 'n'), (5, 5, 4, 'g')]

zz=zip()
print(list(zz))#[]

a=[1,2,3]
zz=zip(a)
print(list(zz))#[(1,), (2,), (3,)]

a=[1,2,3]
b=[1,2,3,4]
c=[1,2,3,4,5]
zz=zip(a,b,c)
print(list(zz))#[(1, 1, 1), (2, 2, 2), (3, 3, 3)]

a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
zz=zip(a,b,c)
print(list(zz))#[(1, 4, 7), (2, 5, 8), (3, 6, 9)]

