from math import sqrt

import numpy as np
import random
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame


def main():
    nodes = input_data('data1.csv')
    nodes.sort()

    generate_centroids(nodes)

    for i in range(0, loop_count):
        membership_of_data_in_clusters(nodes)
        update_centroids(nodes)

    costs.append(cost_calculate(nodes))


def input_data(path):
    data = pd.read_csv(path)
    test_data = DataFrame(data).head(100)
    nodes = []

    for d in (data.iterrows()):
        nodes.append((d[1].values[0], d[1].values[1]))
        # nodes.append(d[1].values[1])

    return nodes


def generate_centroids(nodes):
    # index = 0
    for i in range(0, number_of_clusters):
        # random_centroid = np.random.rand()

        random_centroid = random.choice(nodes)
        centroids.append(random_centroid)
    centroids.sort()

    # random_centroid = random.choice(nodes[index + 1:])
    # index = nodes.index(random_centroid)
    # centroids.append(random_centroid)

    print(centroids)


def membership_of_data_in_clusters(nodes):
    for i in range(0, len(nodes)):
        for j in range(0, number_of_clusters):

            if nodes[i] == centroids[j]:
                mem_amount = 1
                membership[(nodes[i], j)] = mem_amount
                make_others_zero(nodes[i], j)
                break
            else:
                sigma = cal_sigma(nodes[i], centroids[j])
                mem_amount = 1 / sigma

            membership[(nodes[i], j)] = mem_amount


def make_others_zero(node, j):
    for i in range(0, number_of_clusters):
        if i != j:
            membership[(node, j)] = 0


def cal_sigma(node, c):
    sigma = 0
    numerator = cal_distance(node, c)

    for j in range(0, number_of_clusters):
        denominator = cal_distance(node, centroids[j])
        if denominator == 0:
            continue
        div_amount = numerator / denominator
        sigma += pow(div_amount, (2 / (m - 1)))

    return sigma


def cal_distance(node, c):
    s = 0
    for i in range(len(node)):
        s += pow(node[i] - c[i], 2)

    return sqrt(s)


def update_centroids(nodes):
    zero_list = []
    for i in range(len(nodes[0])):
        zero_list.append(0)

    for j in range(0, number_of_clusters):

        numerator = tuple(zero_list)
        denominator = 0
        # print(membership.keys())
        for i in range(0, len(nodes)):
            # print(nodes[i] in membership.values())
            # print(membership[(nodes[i], j)])
            power = pow(membership[(nodes[i], j)], m)
            amount = vector_mul(nodes[i], power)
            numerator = vector_add(numerator, amount)
            denominator += pow(membership[(nodes[i], j)], m)
            # print(pow(membership[(nodes[i], j)], m))

        centroids[j] = vector_div(numerator, denominator)
        # print(numerator, denominator)
        # print('Cj:', centroids[j])
        # print(numerator)


def vector_add(node1, node2):
    new_node = []
    for i in range(len(node1)):
        new_node.append(node1[i] + node2[i])
    return tuple(new_node)


def vector_mul(node, x):
    new_node = []
    for i in range(len(node)):
        new_node.append(node[i] * x)
    return tuple(new_node)


def vector_div(node, x):
    new_node = []
    for i in range(len(node)):
        new_node.append(node[i] / x)
    return tuple(new_node)


def cost_calculate(nodes):
    cost = 0
    for i in range(len(nodes)):
        for j in range(number_of_clusters):
            cost += pow(membership[(nodes[i], j)], m) * pow(cal_distance(nodes[i], centroids[j]), 2)
    return cost


if __name__ == '__main__':
    costs = []
    m = 2
    loop_count = 100
    number_of_clusters = 0

    for c_count in range(0, 1):
        number_of_clusters += 1
        centroids = []
        membership = {}

        main()

    print(costs)
    plt.plot(costs)
    plt.show()
