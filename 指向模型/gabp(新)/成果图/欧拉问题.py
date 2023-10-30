import math

def func(num):
    sum = 0
    for i in range(1, num):
        sum += 1 / math.pow(i, 2)
    return sum
if __name__ == '__main__':
    a = func(10000)
    b = math.pow(math.pi, 2)/6
    print(a)
    print(b)





