import numpy as np

size = 255/4
x = np.arange(0,255,size)
x=np.around(x)
np.append(x,255)
#x[-1] = 255
print(x)

t = np.array([])
print (t)
t = np.append(t, 5)
t = np.append(t,987)
print(t)
print(t.dtype)
print(x.dtype)

my_list = []

my_list.append(12)
my_list.append(4)
print(my_list[1])

arr = np.array([2,4,77,5,99,5,60])
sum = ((25 <= arr) & (arr <=100)).sum()
print(sum)

new_arr = np.zeros_like(arr)
new_arr[((25 <= arr) & (arr <=100))] = -22
print(new_arr)

def changeVar(y):
    y[0]=-289

def pract():
    x=np.zeros_like(t)
    changeVar(x)
    print(x)
    pass

pract()

arr2 = np.array([])
arr1 =arr2