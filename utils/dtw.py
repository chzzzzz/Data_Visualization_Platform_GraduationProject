import pandas as pd
import numpy as np
import math
from numpy import mat
import matplotlib.pyplot as plt#画图
from sklearn.preprocessing import MinMaxScaler#调用了scikit-learn对象MinMaxScaler规范化数据集


# #### 输入处理
# #### Sd 直接距离矩阵

#计算欧几里得距离
def embedding_distance(feature_1, feature_2):
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist

#直接距离矩阵
#复杂度o(n平方)
#只可单维
#这个函数做出了直接距离矩阵Sd

#此函数用来计算直接邻居矩阵
#计算9条数据为一个序列的矩阵之间的欧氏距离，两个矩阵得出一个距离是一个一维数字
def EudistanceMartix_distance(filepath,martix,a):
    #martix矩阵里应当包含一维的时间序列
    #filename = r'D:\chz的大学生活\a大三短学期\光伏发电\data.xls'
    s = ["时间","负载电流", "负载电压","逆变电流", "逆变电压", "输入电流","输入电压"]
    input = pd.read_excel(filepath)[s]
    input = input.values

    scaler = MinMaxScaler(feature_range=(0,1))#对数据进行归一化
    scaler = scaler.fit(input)
    input = scaler.transform(input)
    #9个点为一组一共199组，每组6个
    w = np.zeros(shape=(199,9,6))
    count = 0
    for i in range(0,199):
        for j in range(0,9):
            if (count < len(input)):
                for k in range(0,8):
                    w[i][j][k] = input[count][k]
                count = count + 1
            else:
                break

    len(input)

    data = mat(np.zeros((a + 1,a + 1)))
    for i in range(0,a + 1):
        data[0,i] = i
        data[i,0] = i
    for i in range(0,a):
        for j in range(i,a):
            for k in range(0,6):
                #计算某一维度的欧氏距离
                distance = embedding_distance(w[i][:,k], w[j][:,k])
                data[i + 1,j + 1] = data[i + 1,j + 1] + distance
                data[j + 1,i + 1] = data[j + 1,i + 1] + distance
    return data

# #获得距离矩阵
# Emartix=EudistanceMartix_distance(w,len(w))
# x=Emartix[1:,1:]
# x=x[28].A[0]
# x-x[29]

Emartix = EudistanceMartix_distance(w,len(w))
Emartix


# #### 对Sd直接距离矩阵的最大最小距离函数(在邻接矩阵中也一样可以用)

# In[14]:


def Get_categorymartixmax(martix):
    #martixdis=[]
    martixdis = martix[1:,1:]
    martixdis = martixdis.A
    for i in range(len(martixdis)):
        d = 0.0
        for j in range(len(martixdis)):
            #最小值为a[i][i]
            if(martixdis[i][j] >= d):
                d = martixdis[i][j]
                d_key1 = (i,j)
        tempi,tempj = d_key1
        #print(tempi,tempj)
        #用来验证是否是最大
        martixdis[tempi][tempj] = -1
    for i in range(len(martixdis)):
        for j in range(len(martixdis)):
            if(martixdis[i][j] != -1):
                martixdis[i][j] = 0
    return martixdis

#求最远距离
distancecategorymartixmax = Get_categorymartixmax(Emartix)

#不要试图删去下面这一行 原因不明 反正删除了Emartix就会G
#Python赋值特性

Emartix = EudistanceMartix_distance(w,len(w))   

#得到最近距离的函数
def Get_categorymartixmin(martix):
    martixdis = martix[1:,1:]
    #matirx转换为数组
    martixdis = martixdis.A
    for i in range(len(martixdis)):
        d = math.inf
        for j in range(len(martixdis)):
            #最小值为a[i][i]
            if(martixdis[i][j] <= d and i != j):
                d = martixdis[i][j]
                d_key2 = (i,j)
        tempi,tempj = d_key2
        # print(tempi,tempj)
        martixdis[tempi][tempj] = 1
    for i in range(len(martixdis)):
        for j in range(len(martixdis)):
            if(martixdis[i][j] != 1):
                martixdis[i][j] = 0
    return martixdis

#求最近距离
distancecategorymartixmin = Get_categorymartixmin(Emartix)
distancecategorymartixmax


# #### Sd' 邻接距离矩阵

# In[5]:


#构建邻接点距离矩阵
#复杂度o(n平方)
#只可单维
#邻居矩阵定义：与A有关的所有矩阵的距离序列
def linjudistanceMartix_distance(martix,a):
    b = martix[1:,1:]
#martix矩阵里应当包含一维的时间序列
    data = mat(np.zeros((a,a)))
    for i in range(0,a):
        data[0,i] = i
        data[i,0] = i
    for i in range(0,a - 1):
        for j in range(i,a - 1):
            # print(i,j)
            #计算i与j的距离
            distance = embedding_distance(b[i].A[0], b[j].A[0])#换成一维的ndarray求距离
            data[i + 1,j + 1] = data[i + 1,j + 1] + distance
            data[j + 1,i + 1] = data[j + 1,i + 1] + distance
    return data

# 还是一样的初始化
Emartix = EudistanceMartix_distance(w,len(w))
'''
def EudistanceMartix_distance(martix,a):
#martix矩阵里应当包含一维的时间序列
    data = mat(np.zeros((a + 1,a + 1)))
    for i in range(0,a + 1):
        data[0,i] = i
        data[i,0] = i
    for i in range(0,a):
        for j in range(i,a):
            for k in range(0,8):
                #计算某一维度的欧氏距离
                distance = embedding_distance(w[i][:,k], w[j][:,k])
                data[i + 1,j + 1] = data[i + 1,j + 1] + distance
                data[j + 1,i + 1] = data[j + 1,i + 1] + distance
    return data
'''
#得到邻接距离矩阵
adjacentmartix = linjudistanceMartix_distance(Emartix,len(Emartix))
adjacentmartix


# In[34]:


x=mat([[1,2],[3,4]])
x.A[1]


# In[16]:


adjacentmartix.shape


# #### 对Sd'继续做最大最小距离操作

# In[6]:


# 最大距离
adjacentcategorymartixmax = Get_categorymartixmax(adjacentmartix)

# 初始化,不然会被覆盖
adjacentmartix = linjudistanceMartix_distance(Emartix,len(Emartix))

#最小距离
adjacentcategorymartixmin = Get_categorymartixmin(adjacentmartix)
adjacentcategorymartixmin


# #### 最终的判断矩阵

# In[7]:


#这个函数可以获得1和-1合在一起的矩阵
def Finalarray(max, min):
    finalarray = np.zeros(min.shape)
    for i in range(len(max[0])):
        for j in range(len(max[0])):
            if(max[i][j] == -1):
                finalarray[i][j] = -1
                #print(i,j)
            if(min[i][j] == 1):
                finalarray[i][j] = 1
                # print(i,j)
    return finalarray

#对已经得到最远最近矩阵进行finalarray
finaladjacentarray = Finalarray(adjacentcategorymartixmax,adjacentcategorymartixmin)
finaldistancearray = Finalarray(distancecategorymartixmax,distancecategorymartixmin)


# #### 奇异点查找(已可视化)

# In[8]:


#都最远
data1x = []
data1y = []

for i in range(len(finaladjacentarray[0])):
    for j in range(len(finaladjacentarray[0])):
        if((finaladjacentarray[i][j] == -1 and finaldistancearray[i][j] == -1)):
            # print(i,j)
            data1x.append(i)
            data1y.append(j)

plt.scatter(data1x,data1y)


# In[9]:


#都最近
data2x=[]
data2y=[]
for i in range(len(finaladjacentarray[0])):
    for j in range(len(finaladjacentarray[0])):
        if((finaladjacentarray[i][j]==1 and finaldistancearray[i][j]==1)):
            # print(i,j)
            data2x.append(i)
            data2y.append(j)

plt.scatter(data2x,data2y)


# In[1268]:


#获得还剩下的点的位置
# def get_set(array):
#     array_set=set()
#     for i in array:
#         for item in i:
#             array_set.add(item) 
#     return array_set


# #### 得到最后结果

# In[10]:


#获得还剩下的点的位置
def get_set(array):
    array_set = set()
    for i in array:
        array_set.add(i) 
    return array_set


# In[11]:


# 初始化c列表
c = [i for i in range(199)]


# In[12]:


# datax=[]
# datay=[]
def get_result(finalarray1, finalarray2,k):
    length = len(finalarray1[0])
    while length>k :
        datax=[]
        datay=[]
        #记录两边都是1的点
        for i in range(len(finalarray1[0])):
            for j in range(len(finalarray1[0])):
                if((finalarray1[i][j]==1 and finalarray2[i][j]==1)):
                    datax.append(i)
                    datay.append(j)
        #将这些记录的点在c中改写
        for i in range(len(datax)):
            if(datax[i]>datay[i]):
                for index,data in enumerate(c):
                    if(data==c[datax[i]]):
                        c[index]=c[datay[i]]
            else:
                for index,data in enumerate(c):
                    if(data==c[datay[i]]):
                        c[index]=datax[i]
                        
                    
        # list1=get_num(c)
        # #判别标准
        length=len(get_set(c))
        #根据新的矩阵判别最近点
        # w_temp=np.zeros(shape=(length,9,8))
        # for i,j in enumerate(list1):
        #     w_temp[i]=w[j]
        # #获得新的finalarray1
        # Emartix=EudistanceMartix_distance(w_temp,len(w_temp))
        # distancecategorymax=get_categorymartixmax(Emartix)
        # Emartix=EudistanceMartix_distance(w_temp,len(w_temp))
        # distancecategorymin=get_categorymartixmin(Emartix)
        # finalarray1=finalarray(distancecategorymax,distancecategorymin)
        # #获得新的finalarray2
        # Emartix=EudistanceMartix_distance(w_temp,len(w_temp))
        # adjacent=linjudistanceMartix_distance(Emartix,len(Emartix))
        # adjacentcategorymax=get_categorymartixmax(adjacent)
        # Emartix=EudistanceMartix_distance(w_temp,len(w_temp))
        # adjacent=linjudistanceMartix_distance(Emartix,len(Emartix))
        # adjacentcategorymin=get_categorymartixmin(adjacent)
        # finalarray2=finalarray(adjacentcategorymax,adjacentcategorymin)


get_result(finaldistancearray,finaladjacentarray,160)
print(c)
print(len(get_set(c)))
        


# In[ ]:





# #### 钱哲昊的

# In[1272]:


def Findonce(array) : 
    res = []
    for i in array:
        if array.count(i) == 1:
            res.append(i)
    return res

c_2 = Findonce(c)


# In[1273]:



# Emartix = EudistanceMartix_distance(w,len(w))

# visited = []

# def GenerateNewOutout(output2,dispersed,matrix):
#     martixdis = matrix[1:,1:]
#     martixdis = martixdis.A
    
#     for nodei in dispersed:
#         min = math.inf
#         for nodej in dispersed:
#             if (martixdis[nodei][nodej] < min and nodei != nodej):#and nodej != 0 and nodej != 158:
#                 min = martixdis[nodei][nodej]
#                 temp1, temp2 = nodei,nodej
#         print(temp1,temp2)#检测是否还有可以聚的点
        
#         if not (temp1 in visited and temp2 in visited):
#             output2[temp1] = output2[temp2]
#             visited.append(temp1)
#             visited.append(temp2)
    
#     return output2

# Output2=GenerateNewOutout(c,c_2,Emartix)
# print(c)
# print(len(get_set(c)))


# ### ( •̀ ω •́ )y

# ##### 获得离散点

# In[1274]:


def Findonce(array) : 
    res = []
    for i in array:
        if array.count(i) == 1:
            res.append(i)
    return list(get_set(res))
def FindTwice(array) : 
    res = []
    for i in array:
        if(array.count(i) ==2):
            res.append(i)
    return list(get_set(res))
def Findthree(array) :
    res = []
    for i in array:
        if(array.count(i) >2 and array.count(i)<=4):
            res.append(i)
    return list(get_set(res))
def Findfour(array) :
    res = []
    for i in array:
        if(array.count(i) >4 and array.count(i)<=8):
            res.append(i)
    return list(get_set(res))
def Findfive(array) :
    res = []
    for i in array:
        if(array.count(i) >8 and array.count(i)<=16):
            res.append(i)
    return list(get_set(res))
def Findsix(array) :
    res = []
    for i in array:
        if(array.count(i) >16 and array.count(i)<=32):
            res.append(i)
    return list(get_set(res))
def Findseven(array) :
    res = []
    for i in array:
        if(array.count(i) >32 and array.count(i)<=64):
            res.append(i)
    return list(get_set(res))


# ##### 建立树类和遍历树

# In[1275]:


class Node(object):
    def __init__(self,left=None,right=None,id=None):
        self.left = left
        self.right = right
        self.id = id
def leaf_traversalone(node: Node, label):
    if node.left == None and node.right == None:
        for i in range(len(c)):
            if(c[i]==node.id):
                c[i]=label
        # c[node.id] = label
    if node.left:
        for i in range(len(c)):
            if(c[i]==node.left.id):
                c[i]=label
        leaf_traversalone(node.left, label)
    if node.right:
        for i in range(len(c)):
            if(c[i]==node.right.id):
                c[i]=label
        leaf_traversalone(node.right, label)


# ##### 全局变量

# In[1276]:


global new_node
new_node=Node()
global nodes
nodes=[Node(id=i) for i in range(199)]
global currentid
currentid=199


# ##### 建立函数

# In[1277]:



def HC(innodes,dispersed,matrix,nowid):
    #获得正常的距离矩阵
    visited=[]
    currentid=nowid
    #由于需要迭代所以在输入时就要去掉序号
    # martixdis = matrix[1:,1:]
    # matrixdis = matrix.A
    #获得两个点
    for nodei in dispersed:
        global templeft
        global tempright
        tempright=Node()
        templeft=Node()
        min=math.inf
        for nodej in dispersed:
            #获取最小距离
            if (matrix[nodei][nodej] < min and nodei != nodej):
                min = matrix[nodei][nodej]
                temp1, temp2 = nodei,nodej
        #这段有问题他是新建了一个node而不是选了老的node
        if (temp1 not in visited and temp2 not in visited):
            for node in innodes:
                if(node.id==temp1):
                    templeft=node
            for node in innodes:
                if(node.id==temp2):
                    tempright=node
                visited.append(temp2)
                visited.append(temp1)
        #阿巴阿巴
        if not(templeft.id==None or tempright.id==None):
            print(templeft.id,tempright.id)
            new_node=Node(left=templeft,right=tempright,id=currentid)
            innodes.append(new_node)
            data_temp=[]  
            for i in range(currentid):
                data_temp.append((((matrix[i][new_node.left.id]**2+matrix[i][new_node.right.id]**2-matrix[new_node.left.id][new_node.right.id]**2/2))/2)**0.5)
            #别问我为什么这里用c不用hstack
            matrix=np.c_[matrix,np.array(data_temp)]
            data_temp.append(0)
            #别问我为什么这里用vstack而不用r_
            matrix=np.vstack((matrix,np.array(data_temp)))
            currentid+=1   
    # matrixdis=np.mat(matrixdis)
    return matrix,innodes,currentid


# In[1278]:


print(currentid)


# #### 得出第一次结果

# In[1279]:


hh=Emartix[1:,1:]
hh=hh.A
c_1 = Findonce(c)
nowid=currentid
hh,nodes,currentid=HC(nodes,c_1,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(c)
print(len(get_set(c)))
print(np.shape(hh))


# In[1280]:


print(nowid)
print(currentid)


# ##### 定义一个有效位数函数

# In[1281]:


def Get_effective(nowid,array):
    res=[]
    for i in array:
        if(i<nowid):
            res.append(i)
    return res


# ##### 查看还剩下多少有效值并进行聚类从而减少其有效值

# In[1282]:


c_1=Get_effective(nowid,Findonce(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_1,hh,nowid)


# In[1283]:


for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
Findonce(c)


# In[1284]:


c_1=Get_effective(nowid,Findonce(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_1,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
Findonce(c)


# In[1285]:


print(nodes[200].left.id,nodes[200].right.id)


# ##### 演示是到没有单个的点为止

# In[1286]:


c_1=Get_effective(nowid,Findonce(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_1,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(Findonce(c))
print(c)
print(len(get_set(c)))


# In[1287]:


print(currentid,nowid)
np.shape(nodes)


# #### 得出第二次结果

# In[1288]:


print(nowid)
print(currentid)


# In[1289]:


c_2 = FindTwice(c)
print(c_2)
print(len(c_2))


# In[1290]:


nowid=currentid
hh,nodes,currentid=HC(nodes,c_2,hh,nowid)
print(nowid,currentid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(c)
print(len(get_set(c)))
print(np.shape(hh))


# ##### 重复第一次的操作

# In[1291]:


c_2=Get_effective(nowid,FindTwice(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_2,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(FindTwice(c))
print(c)
print(len(get_set(c)))


# In[1292]:


c_2=Get_effective(nowid,FindTwice(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_2,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(FindTwice(c))
print(c)
print(len(get_set(c)))


# In[1293]:


c_2=Get_effective(nowid,FindTwice(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_2,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(FindTwice(c))
print(c)
print(len(get_set(c)))


# In[1294]:


c_2=Get_effective(nowid,FindTwice(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_2,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(FindTwice(c))
print(c)
print(len(get_set(c)))


# #### 第三次聚类的结果

# In[1295]:


c_3 = Findthree(c)
print(c_3)


# In[1296]:


c_3 = Findthree(c)
nowid=currentid
hh,nodes,currentid=HC(nodes,c_3,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(c)
print(len(get_set(c)))
print(np.shape(hh))


# In[1297]:


print(nowid)
print(currentid)


# In[1298]:


c_3=Get_effective(nowid,Findthree(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_3,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(Findthree(c))
print(c)
print(len(get_set(c)))


# In[1299]:


c_3=Get_effective(nowid,Findthree(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_3,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(Findthree(c))
print(c)
print(len(get_set(c)))


# #### 第四次聚类的结果

# In[1300]:


c_4 = Findfour(c)
nowid=currentid
hh,nodes,currentid=HC(nodes,c_4,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(c)
print(len(get_set(c)))
print(np.shape(hh))
len(c)


# In[1301]:


c_4=Get_effective(nowid,Findfour(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_4,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(Findfour(c))
print(c)
print(len(get_set(c)))


# In[1302]:


c_4=Get_effective(nowid,Findfour(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_4,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(Findfour(c))
print(c)
print(len(get_set(c)))


# #### 第五次聚类的结果

# In[1303]:


c_5 = Findfive(c)
nowid=currentid
hh,nodes,currentid=HC(nodes,c_5,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(c)
print(len(get_set(c)))
print(np.shape(hh))


# In[1304]:


c_5=Get_effective(nowid,Findfive(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_5,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(Findfive(c))
print(c)
print(len(get_set(c)))


# #### 第六次聚类

# In[1305]:


c_6 = Findsix(c)
nowid=currentid
hh,nodes,currentid=HC(nodes,c_6,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(c)
print(len(get_set(c)))
print(np.shape(hh))


# In[1306]:


c_6=Get_effective(nowid,Findsix(c))
nowid=currentid
hh,nodes,currentid=HC(nodes,c_6,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(Findsix(c))
print(c)
print(len(get_set(c)))


# In[1307]:


c_7 = Findseven(c)
nowid=currentid
hh,nodes,currentid=HC(nodes,c_7,hh,nowid)
for i in range(nowid,len(nodes)):
    leaf_traversalone(nodes[i],i)
print(c)
print(len(get_set(c)))
print(np.shape(hh))


# #### 结果

# In[1308]:


for i in get_set(c):
    print(i,c.count(i))


# In[1309]:


for i in range(len(c)):
    if(c[i]==334):
        c[i]=0
    if(c[i]==311):
        c[i]=3
    if(c[i]==319):
        c[i]=2
    if(c[i]==332):
        c[i]=1


# In[1310]:


def hanbao(node: Node, label):
    if node.left ==None and node.right == None:
        print(node.id,"憨包")
        # c[node.id] = label
    if node.left:
        hanbao(node.left, label)
    if node.right:
        hanbao(node.right, label)
hanbao(nodes[334],334)


# In[1311]:


output=[]
#正常类(1:331)(495:867)(1066:1491)，
#故障1(332:494)，
#故障2(868:1065)
#故障3(1492:1791)

for i in range(0,331):
    output.append(0)
for i in range(331,494):
    output.append(1)
for i in range(494,866):
    output.append(0)
for i in range(866,1064):
    output.append(2)
for i in range(1064,1491):
    output.append(0)
for i in range(1491,1791):
    output.append(3)


# In[1312]:


c_final=[]
for i in range(len(c)):
    for j in range(9):
        c_final.append(c[i])


# In[1313]:


sum=0
for i in range(len(c_final)):
    if(c_final[i]==output[i]):
        sum+=1
print(sum)
print(sum/1791)


# In[1314]:


from sklearn.metrics import classification_report,adjusted_rand_score,normalized_mutual_info_score,fowlkes_mallows_score,mutual_info_score,f1_score
score1=classification_report(output,c_final)
print(score1)


# In[1320]:


score2=adjusted_rand_score(output,c_final)
score3=normalized_mutual_info_score(output,c_final)
score4=fowlkes_mallows_score(output,c_final)
score5=mutual_info_score(output,c_final)
score5=f1_score(output,c_final,average='micro')
print(score2)
print(score3)
print(score4)
print(score5)

