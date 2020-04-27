import numpy as np
from random import randint


# 加权表决用于后面仿真过程中，判断任务正确答案。
# 思想就是各自用户可靠性加载0答案权重中，
def average_voting(graph):
    values_mv = {}      # 加权选举的值集合

    for task in graph.tasks:
        g_count = {}
        for u in task.users:                   # 该任务的所有用户
            if u.answer[task] in g_count:          # 统计该任务的所有答案集，但是我们这个设计只有0和1   那么answer在这里不是成绩，而是answer
                g_count[u.answer[task]] += u.reliability[task.kind]  # 加上用户关于该类任务的用户可靠性
            else:
                g_count[u.answer[task]] = u.reliability[task.kind]

        max_count = 0
        max_answer = 0
        for key in g_count:
            if g_count[key] > max_count:
                max_count = g_count[key]
                max_answer = key
        data = []
        data.append(max_answer)
        data.append(max_count / 39)

        values_mv[task] = data    # 存的键是task对象
    return values_mv
