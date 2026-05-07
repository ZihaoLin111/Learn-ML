import random


ans = random.randint(1, 100)

while True:
    guess = int(input("请输入一个1-100的整数"))
    if guess < ans:
        print("小了")
    elif guess > ans:
        print("大了")
    else:
        print("猜对了")
        break
