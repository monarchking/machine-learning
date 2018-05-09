#_*_coding:utf-8_*_
from node import Node
import operator
import math
def ID3(data,defalut):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  dataSet,lableSet=dataLoad(data)
  tree=create_tree(dataSet,lableSet)
  return tree
def prune(tree,testVec):
  '''
  Takes in a trained tree and a validation set of examples.  Prunes nodes in order
  to improve accuracy on the validation data; the precise pruning strategy is up to you.
  '''
  data, labels = dataLoad(testVec)
  node = Node(tree)
  root = node.label
  nextNode = node.children
  index = labels.index(root)
  newTree = {root: {}}
  del (labels[index])
  for key in nextNode.keys():

    if (isinstance(nextNode[key], dict)):
      A = getLablesByfeature_1(data, index, key)
      count = []

      getCount(nextNode[key], A, labels[:], count)
      allnum = 0
      errornum = 0
      for i in count:
        allnum += i[0] + i[1]
        errornum += i[1]
      if (errornum == 0):

        newTree[root][key] = nextNode[key]
        continue

      old = errornum + len(count) * 0.5

      p = old / 1.0 / allnum
      try:
        S = math.sqrt(allnum * p * (1 - p))
      except:
        S=1

      new = errornum + 0.5
      if old - S > new:

        classList = [item[-1] for item in A]
        newTree[root][key] = majorityCnt(classList)
      else:

        subLabel=labels[:]
        subLabel.append('Class')

        lst=[]
        for Data in A:
            B={}
            for j in range(len(subLabel)):
                B[subLabel[j]]=Data[j]
            lst.append(B)
        newTree[root][key] = prune(nextNode[key],lst)
    else:
      newTree[root][key] = nextNode[key]
  return newTree




def test(tree, testVec):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  dataSet = []
  for i in testVec:
      Data = []
      lable = []
      for key, value in i.items():
          Data.append(value)
      dataSet.append(Data)
  classfeature=[example[-1] for example in dataSet]
  count=0
  num=0
  a=[]
  for i in testVec:
      b=i.copy()
      a.append(b)
  for i in a:
      i.pop('Class')
      predictClass=evaluate(tree,i)
      if predictClass==classfeature[num]:
          count+=1
          num+=1
      else:
          num+=1
          continue
  acc=float(count)/len(classfeature)

  return round(acc,3)
def evaluate(tree,testVec):
    '''
    Takes in a tree and one example.  Returns the Class value that the tree
    assigns to the example.
    '''
    if type(tree)!=dict:
      classLable=tree
      return classLable
    else:
      featLables=[]
      for key in testVec.keys():
        featLables.append(key)
      firstStry=list(tree.keys())[0]
      secondDict=tree[firstStry]
      for key in secondDict.keys():
        if testVec[firstStry]==key:
          if type(secondDict[key]).__name__=='dict':
            classLable=evaluate(secondDict[key],testVec)
          else:
            classLable=secondDict[key]
          return classLable
# create tree
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def create_tree(dataSet,lableSet):
  classList = [example[-1] for example in dataSet]
  if classList.count(classList[-1]) == len(classList):
    return classList[-1]
  elif len(lableSet) == 0:
    return majorityCnt(classList)
  bestFeat = findBestSplit(dataSet)
  bestFeatLable = lableSet[bestFeat]
  myTree = {bestFeatLable: {}}
  featValues = [example[bestFeat] for example in dataSet]
  unique_Values = set(featValues)
  del (lableSet[bestFeat])
  for value in unique_Values:
      subData=splitDataSet(dataSet,bestFeat,value)
      subLable=lableSet[:]
      myTree[bestFeatLable][value]=create_tree(subData,subLable)
  return myTree
def splitDataSet(dataSet,axis,value):
  retDataSet=[]
  for featVec in dataSet:
      if featVec[axis]==value:
        reduceFeatVec=featVec[:axis]
        reduceFeatVec.extend(featVec[axis+1:])
        retDataSet.append(reduceFeatVec)
  return retDataSet
def findBestSplit(dataSet):
  # 计算熵
  #print len(dataSet[0])
  numCounts = len(dataSet)
  lables = []
  for i in dataSet:
    lables.append(i[-1])
  lables = list(lables)
  labelCounts = {}
  for i in lables:
    labelCounts[i] = 0
  for featVec in dataSet:
    # 提取出类别变量
    currentLable = featVec[-1]
    if currentLable not in labelCounts.keys():
      labelCounts[currentLable] = 0
    else:
      labelCounts[currentLable] += 1
  # print labelCounts
  shannonEnt = 0.0
  for key in labelCounts:
    prob = float(labelCounts[key]) / numCounts
    # print prob
    shannonEnt -= prob * math.log(prob, 2)

  # print shannonEnt
  # 计算条件熵
  lable_clounm = []
  i = len(dataSet[0]) - 1
  ce_lst = []
  for data_Set in dataSet:
    lable_clounm.append(data_Set[i])
  for i in range(len(dataSet[0]) - 1):
    clounm_data = []
    numCount = {}
    for data_Set in dataSet:
      clounm_data.append(data_Set[i])
    class_data = set(clounm_data)
    clounm_data = list(clounm_data)
    for i in clounm_data:
      numCount[i] = 0
    for featVec in clounm_data:
      currentLable = featVec
      if currentLable not in numCount.keys():
        numCount[currentLable] = 0
      else:
        numCount[currentLable] += 1
    ce = 0.0
    for key in numCount:
      lable1 = []
      label_Counts = {}
      counts = [i for i in range(len(clounm_data)) if clounm_data[i] == key]
      for j in counts:
        lable1.append(lable_clounm[j])

      #
      for i in lable1:
        label_Counts[i] = 0
      for featVec in lable1:
        currentLable = featVec
        if currentLable not in label_Counts.keys():
          label_Counts[currentLable] = 0
        else:
          label_Counts[currentLable] += 1

      shannon_Ent = 0.0
      for key, value in label_Counts.items():
        prob1 = float(value) / len(lable1)
        shannon_Ent -= prob1 * math.log(prob1, 2)
      ce += (float(len(counts)) / len(lable_clounm)) * shannon_Ent

    ce_lst.append(ce)
  # 计算信息增益

  infoGain_lst = []
  for ce in ce_lst:
    newEntropy = ce
    infoGain = shannonEnt - newEntropy
    infoGain_lst.append(infoGain)
  # 选择最好的特征
  index_lst = []
  bestInfo_Gain = sorted(infoGain_lst)
  for i in bestInfo_Gain:
    j = infoGain_lst.index(i)
    infoGain_lst[j]=3
    index_lst.append(j)
  return index_lst[-1]
def dataLoad(data):
  # 加载数据并将其分为标签名列表 ，纯特征类别列表
  dataSet = []
  lableSet = []
  for i in data:
    Data = []
    lable = []
    for key, value in i.items():
      Data.append(value)
      lable.append(key)
    dataSet.append(Data)
  lableSet.append(lable)
  lableSet[0].pop()
  return dataSet,lableSet[0]
def getLablesByfeature_1(traindata, index, feature):
    '''
    通过特征来获取对应的Lables，例如：
    获取school=0,多对应的Lables [0,0,1,0]
    '''
    A = []
    for item in traindata:
        if item[index] == feature:
            temp = item[:index]  # 抽取除index特征外的所有的记录的内容
            temp.extend(item[index + 1:])
            A.append(temp)
    return A
def getCount(tree, data, lables, count):
    node =Node(tree)
    root = node.label
    nextNode = node.children  #对当前节点的列名
    index = lables.index(root)
    del(lables[index])
    for key in nextNode.keys():
        rightcount = 0
        wrongcount = 0
        A = getLablesByfeature_1(data, index, key)
        # 判断是否是叶子节点，不是则迭代进入下一层
        if(isinstance(nextNode[key], dict)):
            getCount(nextNode[key], A, lables[:], count)
        else:
            for item in A:
                # 判断数组给定的分类是否与叶子节点的值相同
                if(str(item[-1]) == str(nextNode[key])):
                    rightcount += 1
                else:
                    wrongcount += 1
            count.append([rightcount, wrongcount])

#print evaluate(tree,dict(a=1, b=1))
#prune_tree=prune(tree,validationData,data)
#print prune_tree

