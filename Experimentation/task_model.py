import numpy as np


# 任务类
class Task:
    def __init__(self, truth, tid):
        self.users = []
        self.id = tid
        self.truth = truth
        self.kind = 0

    def add_user(self, user):
        self.users.append(user)

