from Experimentation import task_model
from Experimentation import user_model
from Experimentation import graph_builder
from Algorithms import MajorityVoting
from Algorithms import AverageVoting

from numpy import genfromtxt
import numpy as np
import csv
from sklearn.cluster import KMeans
import pandas as pd
import copy
import matplotlib.pyplot as plt


alpha = 0.8
# 分的主题数量
kindNum = 4


# 读取数据集 四列分别是[task_id, user_id, answer, truth]
# filename_blue = '../Data/Bluebird/Bluebird.csv'
# 换上我们自生成的数据集
filename_blue = '../Data/Bluebird/newuser.csv'

data_blue = genfromtxt(filename_blue, delimiter=',', dtype=None)

# user hash map
# 键是uid 值是task_id:answer 值也是放成字典合适
user_hm = {}
for record in data_blue:
    if record[1] in user_hm:
        user_hm[record[1]][record[0]] = record[2]
    else:
        user_hm[record[1]] = {}     # 每一个用户对任务的回答集合
        user_hm[record[1]][record[0]] = record[2]

# task hash map
# 键是tid 值是(truth)
task_hm = {}
for record in data_blue:
    if record[0] not in task_hm:
        task_hm[record[0]] = record[3]  # 给每个任务记录上正确答案


users = [user_model.User(u) for u in user_hm]
tasks = [task_model.Task(task_hm[tid], tid) for tid in task_hm]

for t in tasks:
    t.truth = task_hm[t.id]


graph = graph_builder.Graph(users, tasks)

for t in tasks:
    for u in users:
        graph.pick_user(t, u.id)


# 将任务答案存放在User类中
for u in users:
    for t in tasks:
        graph.pick_task(u, t.id, user_hm[u.id][t.id])

# for u in users:
#     for t in tasks:
#         print(t.id, u.answer[t])

# 以上完成初始化行为
# 先做聚类操作，把任务划成两类
# 将任务以及所有用户答案写成csv文件
data = [[0] * len(users) for _ in range(len(tasks))]
# data = [[]]
for t in tasks:
    for u in users:
        data[t.id][u.id] = user_hm[u.id][t.id]

with open("../Data/Bluebird/data.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

# 参数初始化
inputfile = '../Data/Bluebird/data.csv'

loan_data = pd.DataFrame(pd.read_csv(inputfile, header=None, sep=','))

# # loan_data = pd.DataFrame(pd.read_csv(inputfile, header=None, names=['task_id', 'user_id', 'answer', 'truth']))
# # loan_data = pd.DataFrame(data_blue)
# print(loan_data.head())

a = []
for i in range(0, 39):
    a.append(i)

loan = np.array(loan_data[a])

clf = KMeans(n_clusters=kindNum)
clf = clf.fit(loan)
# 展示质心
# print(clf.cluster_centers_)

# 测试预测结果
# test = [[]]
# for i in range(0, 39):
#     test[0].append(0)
#
# print(clf.predict(test))

# 所属类别
loan_data['label'] = clf.labels_
loan_data_array = loan_data.values
# print(loan_data.values)

# 分好类了
for t in tasks:
    t.kind = loan_data_array[t.id][len(users)]

gold_tasks = tasks
# 计算初始可靠性
for u in users:
    for kind in range(kindNum):
        count = 0
        right = 0
        for t in gold_tasks:
            if t.kind == kind:
                count += 1
                if u.answer[t] == t.truth:
                    right += 1

        u.reliability[kind] = right/count
        # print(u.id, kind, right)


# print("展示每个用户的各类的可靠度")
# for u in users:
#     print(u.id)
#     for kind in range(kindNum):
#         print(kind, u.reliability[kind])
#     print()

# 多数表决和加权表决的结果展示
show_data = []

# 测试一下多数选举和正确答案的异同
majority_res = MajorityVoting.majority_voting(graph)
right_count = 0
for t in tasks:
    if t.truth == majority_res[t][0]:
        right_count += 1
# 多数选举的正确率在0.76左右
majority_accuracy = right_count / 108

# print("多数选举的正确率为： ", majority_accuracy)

#
# # 测试加权选举和正确答案的异同
average_res = AverageVoting.average_voting(graph)
# right_count = 0
# for t in tasks:
#     if t.truth == average_res[t][0]:
#         right_count += 1
# # 多数选举的正确率在0.76左右
# average_accuracy = right_count / 108
# # print("加权选举的正确率为： ", average_accuracy)
#
# data = []
# data.append(majority_accuracy)
# data.append(average_accuracy)
# show_data.append(data)
# print(show_data[0])
#
#
# 更新可靠性过程, 更新0到72任务
test_tasks = tasks[0: 108]
for u in users:
    for kind in range(kindNum):
        for t in test_tasks:
            # 处在对应类，才开始更新
            if t.kind == kind:
                # 这里成绩用多数选举的正确率吧
                grade = 0
                if u.answer[t] == average_res[t][0]:
                    # grade = majority_res[t][1]
                    grade = 1
                # 更新
                u.reliability[kind] = u.reliability[kind] * alpha + (1 - alpha) * grade

# # 测试一下多数选举和正确答案的异同
# majority_res = MajorityVoting.majority_voting(graph)
# right_count = 0
# for t in tasks:
#     if t.truth == majority_res[t][0]:
#         right_count += 1
# # 多数选举的正确率在0.76左右
# majority_accuracy = right_count / 108
#
# # 测试加权选举和正确答案的异同
# average_res = AverageVoting.average_voting(graph)
# right_count = 0
# for t in tasks:
#     if t.truth == average_res[t][0]:
#         right_count += 1
# # 多数选举的正确率在0.76左右
# average_accuracy = right_count / 108
# # print("加权选举的正确率为： ", average_accuracy)
# # 添加训练0-72任务的结果
# data = []
# data.append(majority_accuracy)
# data.append(average_accuracy)
# show_data.append(data)
# print(show_data[1])
#
#
# 更新可靠性过程, 更新0到72任务
# test_tasks = tasks[0: 108]
# for u in users:
#     for kind in range(kindNum):
#         for t in test_tasks:
#             # 处在对应类，才开始更新
#             if t.kind == kind:
#                 # 这里成绩用多数选举的正确率吧
#                 grade = 0
#                 if u.answer[t] == majority_res[t][0]:
#                     # grade = majority_res[t][1]
#                     grade = 1
#                 # 更新
#                 u.reliability[kind] = u.reliability[kind] * alpha + (1 - alpha) * grade

# # 测试一下多数选举和正确答案的异同
# majority_res = MajorityVoting.majority_voting(graph)
# right_count = 0
# for t in tasks:
#     if t.truth == majority_res[t][0]:
#         right_count += 1
# # 多数选举的正确率在0.76左右
# majority_accuracy = right_count / 108
#
# # 测试加权选举和正确答案的异同
# average_res = AverageVoting.average_voting(graph)
# right_count = 0
# for t in tasks:
#     if t.truth == average_res[t][0]:
#         right_count += 1
# # 多数选举的正确率在0.76左右
# average_accuracy = right_count / 108
# # print("加权选举的正确率为： ", average_accuracy)
# # 添加训练0-72任务的结果
# data = []
# data.append(majority_accuracy)
# data.append(average_accuracy)
# show_data.append(data)
# print(show_data[2])

# with open("../Result/voting.csv", "w") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(show_data)

# 现在是画多数表决和加权表决的图
# 条形图的绘制--水平交错条形图
# 导入第三方模块


# # 后面方便造数据了
# # # 读入数据
# result = pd.read_excel('../Result/voting.xlsx')
# # 取出城市名称
# # Cities = HuRun.City.unique()
# name = result.Name
# major = result.Major
# aver = result.Aver
#
# # 绘制水平交错条形图
# bar_width = 0.4
# plt.bar(x = np.arange(len(name)), height = major, label = 'Major', color = 'steelblue', width = bar_width)
# plt.bar(x = np.arange(len(name))+bar_width, height = aver, label = 'Average', color = 'indianred', width = bar_width)
# # 添加刻度标签（向右偏移0.225）
# plt.xticks(np.arange(5)+0.2, name)
# # 添加y轴标签
# plt.ylabel('Voting accuracy')
# # 添加图形标题
# # plt.title('多数表决与加权表决的比较')
# # 添加图例
# plt.legend()
# # 显示图形
# plt.show()













# 更新完成后，用每类别的信赖度作为用户聚类的衡量属性
# 将每个用户的每个类别的可靠性写成csv文件

# user_reliability = [[0] * kindNum for _ in range(39)]
# for u in users:
#     for kind in range(kindNum):
#         user_reliability[u.id][kind] = u.reliability[kind]

# 试试看用均值版本
user_reliability = [[0] * 2 for _ in range(len(users))]
for u in users:
    user_sum = 0

    for i in range(kindNum // 2):
        user_reliability[u.id][0] += u.reliability[i]

    for i in range(kindNum//2, kindNum):

        user_reliability[u.id][1] += u.reliability[i]
    user_reliability[u.id][0] = user_reliability[u.id][0] / (kindNum//2)

    user_reliability[u.id][1] = user_reliability[u.id][1] / (kindNum - (kindNum // 2))

# 再试试看，种类占比加权的均值版本


with open("../Data/Bluebird/user_reliability.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(user_reliability)

inputfile = '../Data/Bluebird/user_reliability.csv'

loan_data = pd.DataFrame(pd.read_csv(inputfile, header=None, sep=','))

a = []
for i in range(2):
    a.append(i)

# 将根据可靠性分类的
# 四分类，选取质心最小的作为异常类


clf = KMeans(n_clusters=4)
loan = np.array(loan_data[a])
clf = clf.fit(loan)
#提取不同类别的数据
loan_data['label'] = clf.labels_
#
# loan_data0 = loan_data.loc[loan_data["label"] == 0]
# loan_data1 = loan_data.loc[loan_data["label"] == 1]
# loan_data2 = loan_data.loc[loan_data["label"] == 2]
# loan_data3 = loan_data.loc[loan_data["label"] == 3]
#
#
#
# plt.scatter(loan_data0[0],loan_data0[1],50,color='#99CC01',marker='+',linewidth=2,alpha=0.8)
# plt.scatter(loan_data1[0],loan_data1[1],50,color='#FE0000',marker='+',linewidth=2,alpha=0.8)
# plt.scatter(loan_data2[0],loan_data2[1],50,color='#0000FE',marker='+',linewidth=2,alpha=0.8)
# plt.scatter(loan_data3[0],loan_data3[1],50,color='#000000',marker='+',linewidth=2,alpha=0.8)
#
# plt.xlabel('kind1 task')
# plt.ylabel('kind2 task')
# plt.xlim(0, 1)
# plt.ylim(0, 1)
# plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)
# plt.show()




# 展示质心
# print("根据用户细粒度可靠性聚类计算的质心")
# print(clf.cluster_centers_)





print("根据聚类做的分类")
good_user = []
bad_user = []



# 不用这种方法，求和也还是会有极端情况的出现，不想让偏科型用户视为异常用户
# 综合判断质心在每类上最小（这里就简单的用和）
barycenter = [0 for _ in range(len(clf.cluster_centers_))]
for i in range(len(clf.cluster_centers_)):
    for j in clf.cluster_centers_[i]:
        barycenter[i] += j
min_center = min(barycenter)
min_index = barycenter.index(min_center)
print("最小的质心", min_index)

loan_data_array = loan_data.values
# print(loan_data_array)

for i in range(39):
    if loan_data_array[i][2] == min_index:
        bad_user.append(i)
    else:
        good_user.append(i)


print("正常用户")
print(good_user)

print("异常用户")
print(bad_user)


print("正常用户数量为:", len(good_user))
print("异常用户数量为:", len(bad_user))




# 根据绝对值计算的异常用户
# 异常用户认定为9：1
# abnormal_user_num = (len(user_hm) // 10 + 1)
abnormal_user_num = len(bad_user)

good_user = []
bad_user = []
# 每个用户可靠性与原点差的平方和
user_reliability_sqrt = [0 for _ in range(len(users))]
for u in users:
    sqrt_sum = 0
    for kind in range(kindNum):
        a = round(u.reliability[kind], 5)
        sqrt_sum += a*a
    user_reliability_sqrt[u.id] = sqrt_sum

sort_sqrt_list = copy.copy(user_reliability_sqrt)
sort_sqrt_list.sort()




for i in range(abnormal_user_num):

    bad_user.append(user_reliability_sqrt.index(sort_sqrt_list[i]))

# for i in range(len(sort_sqrt_list)):
#     print(user_reliability_sqrt.index(sort_sqrt_list[i]), sort_sqrt_list[i])

for u in users:
    if u.id not in bad_user:
        good_user.append(u.id)




good_user.sort()
bad_user.sort()

print("正常用户")
print(good_user)

print("异常用户")
print(bad_user)

print("正常用户数量为:", len(good_user))
print("异常用户数量为:", len(bad_user))


# 计算不分类的正确率
correct = {}
for u in users:
    right = 0
    for t in tasks:
        if u.answer[t] == t.truth:
            right += 1
    correct[u.id] = right/108

score = []
for key in correct.keys():
    # print(key, correct[key])
    score.append(correct[key])

score.sort()

# for i in range(abnormal_user_num):
#     for key in correct.keys():
#         if correct[key] == score[i]:
#             print(key)


# 这里做去除异常用户前后正确率的比较，任务分配是维护一个用户主题队列。
# worker_capacity = [1,2,3,4,5,6]

worker_capacity = [10, 20, 30, 40, 50, 60]
kind2_user = []
kind1_user = []
kind3_user = []
kind0_user = []

for user in users:
    kind1_user.append(user)
    kind2_user.append(user)
    kind3_user.append(user)
    kind0_user.append(user)

# 好了这是待选任务者
kind0_user = sorted(kind0_user, key=lambda z: z.reliability[0], reverse=True)
kind1_user = sorted(kind1_user, key=lambda user: user.reliability[1], reverse=True)
kind2_user = sorted(kind2_user, key=lambda user: user.reliability[2], reverse=True)
kind3_user = sorted(kind3_user, key=lambda user: user.reliability[3], reverse=True)



capacity0 = 0
capacity1 = 0
capacity2 = 0
capacity3 = 0

task = []
for i in range(10):
    for t in tasks:
        task.append(t)

before_accuracy = []
# 带异常对象的集合
idx = 0
right_count = 0
for item in task:
    if item.kind == 0:
        if capacity0 + 1 > worker_capacity[idx]:
            temp = kind0_user[0]
            kind0_user.remove(temp)
            kind0_user.append(temp)

            capacity0 = 0

        if kind0_user[0].answer[item] == item.truth:
            right_count += 1
        capacity0 += 1
    if item.kind == 1:
        if capacity1 + 1 > worker_capacity[idx]:
            temp = kind1_user[1]
            kind1_user.remove(temp)
            kind1_user.append(temp)
            capacity1 = 0
        if kind1_user[0].answer[item] == item.truth:
            right_count += 1
        capacity1 += 1
    if item.kind == 2:
        if capacity2 + 1 > worker_capacity[idx]:
            temp = kind2_user[0]
            kind2_user.remove(temp)
            kind2_user.append(temp)
            capacity2 = 0
        if kind2_user[0].answer[item] == item.truth:
            right_count += 1
        capacity2 += 1
    if item.kind == 3:
        if capacity3 + 1 > worker_capacity[idx]:
            temp = kind3_user[0]
            kind3_user.remove(temp)
            kind3_user.append(temp)
            capacity3 = 0
        if kind3_user[0].answer[item] == item.truth:
            right_count += 1
        capacity3 +=1

before_accuracy.append(right_count/len(task))

kind0_user = sorted(kind0_user, key=lambda z: z.reliability[0], reverse=True)
kind1_user = sorted(kind1_user, key=lambda user: user.reliability[1], reverse=True)
kind2_user = sorted(kind2_user, key=lambda user: user.reliability[2], reverse=True)
kind3_user = sorted(kind3_user, key=lambda user: user.reliability[3],reverse=True)

# 带异常对象的集合
idx = 1
right_count = 0
for item in task:
    if item.kind == 0:
        if capacity0 + 1 > worker_capacity[idx]:
            temp = kind0_user[0]
            kind0_user.remove(temp)
            kind0_user.append(temp)

            capacity0 = 0

        if kind0_user[0].answer[item] == item.truth:
            right_count += 1
        capacity0 += 1
    if item.kind == 1:
        if capacity1 + 1 > worker_capacity[idx]:
            temp = kind1_user[1]
            kind1_user.remove(temp)
            kind1_user.append(temp)
            capacity1 = 0
        if kind1_user[0].answer[item] == item.truth:
            right_count += 1
        capacity1 += 1
    if item.kind == 2:
        if capacity2 + 1 > worker_capacity[idx]:
            temp = kind2_user[0]
            kind2_user.remove(temp)
            kind2_user.append(temp)
            capacity2 = 0
        if kind2_user[0].answer[item] == item.truth:
            right_count += 1
        capacity2 += 1
    if item.kind == 3:
        if capacity3 + 1 > worker_capacity[idx]:
            temp = kind3_user[0]
            kind3_user.remove(temp)
            kind3_user.append(temp)
            capacity3 = 0
        if kind3_user[0].answer[item] == item.truth:
            right_count += 1
        capacity3 +=1

before_accuracy.append(right_count/len(task))

kind0_user = sorted(kind0_user, key=lambda z: z.reliability[0], reverse=True)
kind1_user = sorted(kind1_user, key=lambda user: user.reliability[1], reverse=True)
kind2_user = sorted(kind2_user, key=lambda user: user.reliability[2], reverse=True)
kind3_user = sorted(kind3_user, key=lambda user: user.reliability[3], reverse=True)

# 带异常对象的集合
idx = 2
right_count = 0
for item in task:
    if item.kind == 0:
        if capacity0 + 1 > worker_capacity[idx]:
            temp = kind0_user[0]
            kind0_user.remove(temp)
            kind0_user.append(temp)

            capacity0 = 0

        if kind0_user[0].answer[item] == item.truth:
            right_count += 1
        capacity0 += 1
    if item.kind == 1:
        if capacity1 + 1 > worker_capacity[idx]:
            temp = kind1_user[1]
            kind1_user.remove(temp)
            kind1_user.append(temp)
            capacity1 = 0
        if kind1_user[0].answer[item] == item.truth:
            right_count += 1
        capacity1 += 1
    if item.kind == 2:
        if capacity2 + 1 > worker_capacity[idx]:
            temp = kind2_user[0]
            kind2_user.remove(temp)
            kind2_user.append(temp)
            capacity2 = 0
        if kind2_user[0].answer[item] == item.truth:
            right_count += 1
        capacity2 += 1
    if item.kind == 3:
        if capacity3 + 1 > worker_capacity[idx]:
            temp = kind3_user[0]
            kind3_user.remove(temp)
            kind3_user.append(temp)
            capacity3 = 0
        if kind3_user[0].answer[item] == item.truth:
            right_count += 1
        capacity3 += 1

before_accuracy.append(right_count/len(task))

kind0_user = sorted(kind0_user, key=lambda z: z.reliability[0], reverse=True)
kind1_user = sorted(kind1_user, key=lambda user: user.reliability[1], reverse=True)
kind2_user = sorted(kind2_user, key=lambda user: user.reliability[2], reverse=True)
kind3_user = sorted(kind3_user, key=lambda user: user.reliability[3], reverse=True)

# 带异常对象的集合
idx = 3
right_count = 0
for item in task:
    if item.kind == 0:
        if capacity0 + 1 > worker_capacity[idx]:
            temp = kind0_user[0]
            kind0_user.remove(temp)
            kind0_user.append(temp)

            capacity0 = 0

        if kind0_user[0].answer[item] == item.truth:
            right_count += 1
        capacity0 += 1
    if item.kind == 1:
        if capacity1 + 1 > worker_capacity[idx]:
            temp = kind1_user[1]
            kind1_user.remove(temp)
            kind1_user.append(temp)
            capacity1 = 0
        if kind1_user[0].answer[item] == item.truth:
            right_count += 1
        capacity1 += 1
    if item.kind == 2:
        if capacity2 + 1 > worker_capacity[idx]:
            temp = kind2_user[0]
            kind2_user.remove(temp)
            kind2_user.append(temp)
            capacity2 = 0
        if kind2_user[0].answer[item] == item.truth:
            right_count += 1
        capacity2 += 1
    if item.kind == 3:
        if capacity3 + 1 > worker_capacity[idx]:
            temp = kind3_user[0]
            kind3_user.remove(temp)
            kind3_user.append(temp)
            capacity3 = 0
        if kind3_user[0].answer[item] == item.truth:
            right_count += 1
        capacity3 += 1

before_accuracy.append(right_count/len(task))

kind0_user = sorted(kind0_user, key=lambda z: z.reliability[0], reverse=True)
kind1_user = sorted(kind1_user, key=lambda user: user.reliability[1], reverse=True)
kind2_user = sorted(kind2_user, key=lambda user: user.reliability[2], reverse=True)
kind3_user = sorted(kind3_user, key=lambda user: user.reliability[3], reverse=True)

# 带异常对象的集合
idx = 4
right_count = 0
for item in task:
    if item.kind == 0:
        if capacity0 + 1 > worker_capacity[idx]:
            temp = kind0_user[0]
            kind0_user.remove(temp)
            kind0_user.append(temp)

            capacity0 = 0

        if kind0_user[0].answer[item] == item.truth:
            right_count += 1
        capacity0 += 1
    if item.kind == 1:
        if capacity1 + 1 > worker_capacity[idx]:
            temp = kind1_user[1]
            kind1_user.remove(temp)
            kind1_user.append(temp)
            capacity1 = 0
        if kind1_user[0].answer[item] == item.truth:
            right_count += 1
        capacity1 += 1
    if item.kind == 2:
        if capacity2 + 1 > worker_capacity[idx]:
            temp = kind2_user[0]
            kind2_user.remove(temp)
            kind2_user.append(temp)
            capacity2 = 0
        if kind2_user[0].answer[item] == item.truth:
            right_count += 1
        capacity2 += 1
    if item.kind == 3:
        if capacity3 + 1 > worker_capacity[idx]:
            temp = kind3_user[0]
            kind3_user.remove(temp)
            kind3_user.append(temp)
            capacity3 = 0
        if kind3_user[0].answer[item] == item.truth:
            right_count += 1
        capacity3 += 1

before_accuracy.append(right_count/len(task))

kind0_user = sorted(kind0_user, key=lambda z: z.reliability[0], reverse=True)
kind1_user = sorted(kind1_user, key=lambda user: user.reliability[1], reverse=True)
kind2_user = sorted(kind2_user, key=lambda user: user.reliability[2], reverse=True)
kind3_user = sorted(kind3_user, key=lambda user: user.reliability[3], reverse=True)

# 带异常对象的集合
idx = 5
right_count = 0
for item in task:
    if item.kind == 0:
        if capacity0 + 1 > worker_capacity[idx]:
            temp = kind0_user[0]
            kind0_user.remove(temp)
            kind0_user.append(temp)

            capacity0 = 0

        if kind0_user[0].answer[item] == item.truth:
            right_count += 1
        capacity0 += 1
    if item.kind == 1:
        if capacity1 + 1 > worker_capacity[idx]:
            temp = kind1_user[1]
            kind1_user.remove(temp)
            kind1_user.append(temp)
            capacity1 = 0
        if kind1_user[0].answer[item] == item.truth:
            right_count += 1
        capacity1 += 1
    if item.kind == 2:
        if capacity2 + 1 > worker_capacity[idx]:
            temp = kind2_user[0]
            kind2_user.remove(temp)
            kind2_user.append(temp)
            capacity2 = 0
        if kind2_user[0].answer[item] == item.truth:
            right_count += 1
        capacity2 += 1
    if item.kind == 3:
        if capacity3 + 1 > worker_capacity[idx]:
            temp = kind3_user[0]
            kind3_user.remove(temp)
            kind3_user.append(temp)
            capacity3 = 0
        if kind3_user[0].answer[item] == item.truth:
            right_count += 1
        capacity3 += 1

before_accuracy.append(right_count/len(task))

kind_user_list = []

kind_user_list.append(kind0_user)
kind_user_list.append(kind1_user)
kind_user_list.append(kind2_user)
kind_user_list.append(kind3_user)


capacity_list = [0, 0, 0, 0]
for u in bad_user:
    for user in users:
        if u == user.id:
            for list in kind_user_list:

                list.remove(user)



after_accuracy = []
for idx in range(6):
    right_count = 0
    kind0_user = sorted(kind0_user, key=lambda z: z.reliability[0], reverse=True)
    kind1_user = sorted(kind1_user, key=lambda user: user.reliability[1], reverse=True)
    kind2_user = sorted(kind2_user, key=lambda user: user.reliability[2], reverse=True)
    kind3_user = sorted(kind3_user, key=lambda user: user.reliability[3], reverse=True)

    for item in task:
        k = item.kind
        if capacity_list[k] + 1 > worker_capacity[idx]:
            temp = kind_user_list[k][0]
            kind_user_list[k].remove(temp)
            kind_user_list[k].append(temp)
            capacity_list[k] = 0
        if kind_user_list[k][0].answer[item] == item.truth:
            right_count += 1
        capacity_list[k] += 1

    after_accuracy.append(right_count/len(task))
# print("去除异常用户之前")
# for u in before_accuracy:
#     print(u)
#
# print("去除了异常用户之后： ")
# for u in after_accuracy:
#     print(u)

# table = pd.read_excel('../Result/remove.xlsx')
#
# print(table)
# capacity = table.capacity
# before = table.before_accuracy.unique()
# after = table.after_accuracy
#
# x = capacity
# y1, y2 = before, after
# plt.plot(x, np.array(y1), marker="o", label="before")
# plt.plot(x, np.array(y2), marker="*", label="after")
# plt.xlabel("user's capacity of tasks")
# plt.ylabel('accuracy')
#
# plt.ylim(0, 1)
# plt.legend()
# plt.show()
#
# import random
#
# # 自生成用户
# new_data = []
#
#
# # yuzhi = [30, 60, 70, 75, 80]
# yuzhi = [30, 45, 60, 75, 90]
#
# # 添加
# for t in tasks:
#     for i in range(0, 20):
#         user = []
#         user.append(t.id)
#         user.append(i)
#
#         temple = random.randint(1, 100)
#         if temple <= yuzhi[0]:
#             user.append(t.truth)
#         else:
#             if t.truth == 1:
#                 user.append(0)
#             else:
#                 user.append(1)
#         user.append(t.truth)
#         new_data.append(user)
#     for i in range(20, 40):
#         user = []
#         user.append(t.id)
#         user.append(i)
#         temple = random.randint(1, 100)
#         if temple <=  yuzhi[1]:
#             user.append(t.truth)
#         else:
#             if t.truth == 1:
#                 user.append(0)
#             else:
#                 user.append(1)
#         user.append(t.truth)
#         new_data.append(user)
#     for i in range(40, 70):
#         user = []
#         user.append(t.id)
#         user.append(i)
#         temple = random.randint(1, 100)
#         if temple <=  yuzhi[2]:
#             user.append(t.truth)
#         else:
#             if t.truth == 1:
#                 user.append(0)
#             else:
#                 user.append(1)
#         user.append(t.truth)
#         new_data.append(user)
#     for i in range(70, 90):
#         user = []
#         user.append(t.id)
#         user.append(i)
#         temple = random.randint(1, 100)
#         if temple <=  yuzhi[3]:
#             user.append(t.truth)
#         else:
#             if t.truth == 1:
#                 user.append(0)
#             else:
#                 user.append(1)
#         user.append(t.truth)
#         new_data.append(user)
#     for i in range(90, 100):
#         user = []
#         user.append(t.id)
#         user.append(i)
#         temple = random.randint(1, 100)
#         if temple <=  yuzhi[4]:
#             user.append(t.truth)
#         else:
#             if t.truth == 1:
#                 user.append(0)
#             else:
#                 user.append(1)
#         user.append(t.truth)
#         new_data.append(user)
#
#
# with open("../Data/Bluebird/newuser.csv", "w") as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(new_data)
#
# TP = 0
# for u in bad_user:
#     if u < 20:
#         TP += 1
#
# precision = TP / len(bad_user)
# recall = TP / 20
# f1 = 2 * recall * precision / (precision + recall)
#
# print("准确率: {0} 回召率： {1} f1: {2}".format(precision,  recall, f1))


# ==========================================================================================
# 画图

table = pd.read_excel('../Result/fscore.xlsx')

print(table)
a = table.a
p = table.p
r = table.r
f = table.f

x = a
y1, y2, y3 = p, r, f
plt.plot(x, np.array(y1), marker="o", label="Precision")
plt.plot(x, np.array(y2), marker="*", label="Recall")
plt.plot(x, np.array(y3), marker="+", label="F-score")

plt.xlabel("Anomalous user's accuracy")
plt.ylabel('Evaluating criteria')
plt.gca().invert_xaxis()
plt.ylim(0, 1)
plt.legend()
plt.show()
