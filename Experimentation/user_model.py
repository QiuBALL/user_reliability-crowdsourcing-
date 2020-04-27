
class User:
    def __init__(self, uid):
        self.tasks = []         # 任务集合
        self.id = uid           # 用户的id
        self.answer = {}        # 对每个任务的答案（字典） （键：任务对象） （值：答案）
        self.reliability = {}   # 用户的可信度(字典） 根据任务类别加的可信度

    # 添加任务
    def add_task(self, task, answer):       
        self.tasks.append(task)
        self.answer[task] = answer
