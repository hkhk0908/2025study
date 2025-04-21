def grade():
    if list1[i] == 100:
        print("100점")

    elif list1[i] == 0:
        print("0점")

    elif list1[i] >= 90:
        print("A")

    elif list1[i] >= 80:
        print("B")

    elif list1[i] >= 70:
        print("C")

    elif list1[i] >= 60:
        print("D")

    else:
        print("F")

def change():
    print(list1)
    x = int(input('몇번째 자리를 바꾸실건가요?: '))
    y = int(input("바꿀 점수를 입력하세요: "))
    list1[x - 1] = y
    print(list1)


print("학생성적처리시스템")
print("================")

list1 = [0,0,0]

while True:
    print("1.성적입력 2.성적출력 3.성적교체 ")
    num = int(input("보기를 선택하세요: "))

    if num == 1:
        for i in range(len(list1)):
            list1[i] = int(input("숫자를 입력하세요: "))

    elif num == 2:
        for i in range(len(list1)):
            grade()

    elif num == 3:
        change()
