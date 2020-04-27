class Graph:
    def __init__(self, users, tasks):
        self.users = users
        self.tasks = tasks
        self.n_tasks = len(tasks)       # 任务数量
        self.n_users = len(users)

        self.user_index_hm = {}         # 建立以user的id为索引的用户hash map
        for u in self.users:
            self.user_index_hm[u.id] = u

        self.task_index_hm = {}          # 建立以task的id为索引的用户hash map
        for it in self.tasks:
            self.task_index_hm[it.id] = it

    # 给任务添加用户
    def pick_user(self, task, uid):
        task.add_user(self.user_index_hm[uid])

    # 给用户添加任务，包括答给任务的答案
    def pick_task(self, user, tid, answer):
        user.add_task(self.task_index_hm[tid], answer)
