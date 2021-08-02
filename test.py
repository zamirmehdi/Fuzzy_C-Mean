from math import sqrt

from matplotlib import pyplot as plt

a = (186.8924713050765, 334.24371319690295, 1)
c = 534.24371319690295
b = (576.8924713050765, c)
print(len(a))
print(sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2)))

a = 100
b = 200

print(sqrt(pow(a - b, 2)))

new_node = []
node = (186.8924713050765, 334.24371319690295, 1)
x = 2
for i in range(len(node)):
    new_node.append(node[i] * x)
print( tuple(new_node))

print(new_node)
plt.plot(new_node)
plt.show()
