print("계산기")
print("=======================")
a = int(input("계산기를 실행시킬려면 1을 누르세요: "))
if a == 1 :
    list1 = []
    while True:
        print("1.더하기 2.빼기 3.곱하기 4.나누기 0.종료")

        b = input("보기를 골라주세요: ")
        b = int(b)
        if b == 0:
            print("종료합니다.")
            break

        x, y = input("두 수를 입력하세요:").split()
        x, y = int(x), int(y)

        result = 0

        if b == 1:
            result = x + y
        elif b == 2:
            result = x - y
        elif b == 3:
            result = x * y
        elif b == 4:
            result = x / y

        list1.append(result)
        print(list1)
else :
    print("숫자를 잘못입력하셨습니다. 종료합니다.")