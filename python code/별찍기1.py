for i in range(0,5):
    print("*" * (i+1))

for i in range(0,5):
    print(' ' * (5-(i+1)),end='')
    print("*" * (i+1))

for i in range(0,5):
    print("*" * (5-i))

for i in range(0,5):
    print(' ' * i,end='')
    print("*" * (5-i))