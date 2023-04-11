def single_digit(num):
    while num >= 10:
        num = sum(int(i) for i in str(num))
    return num

print(single_digit(1234))