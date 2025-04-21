for i in range(0,5):
    print(' ' * (5-(i+1)),end='')
    print('*' * (2 * i + 1))

for i in range(5,0,-1):
    print(' ' * (5-i),end='')
    print('*' * (2 * i - 1))

for i in range(1,5):
    print(" "* (5-(i+1)),end='')
    print("*"* (2 * i + 1))
for j in range(4, 0, -1):
    print(" "* (5-j),end='')
    print("*"* (2 * j - 1))

for i in range(0,5):
    print(" " * (5-i)*4,end='')
    print("*"* (8 * i+1))
for j in range(5,0,-1):
    print(" " * (5-j)*4,end='')
    print("*"* (8 * j+1))
print(" " * 20 + "*")