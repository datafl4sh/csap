import math

bal = 1.71828182845904523536028747135

i = 1
while i <= 25:
    bal = i*bal - 1
    i = i + 1

print("Balance after 25 years: ${} ( => ${:2.2e})".format(bal,bal))

