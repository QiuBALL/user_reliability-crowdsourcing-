import numpy as np
from random import randint


# 多数投票用于后面仿真过程中，判断任务正确答案。
def majority_voting(graph):
    values_mv = {}      # 多数选举的值集合

    for task in graph.tasks:
        g_count = {}
        for u in task.users:                   # 该任务的所有用户
            if u.answer[task] in g_count:          # 统计该任务的所有答案集，但是我们这个设计只有0和1   那么answer在这里不是成绩，而是answer
                g_count[u.answer[task]] += 1
            else:
                g_count[u.answer[task]] = 1

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
