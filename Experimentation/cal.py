# while True:
#     a = float(input())
#     b = float(input())
#     c = float(input())
#
#     print((a+b+c)/ 3)
recall = 0.867
precision = 0.746

f1 = 2 * recall * precision / (precision + recall)
print (f1)