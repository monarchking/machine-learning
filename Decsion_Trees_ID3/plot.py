import parse
import random
import ID3

x_axis = []
with_prune = []
without_prune=[]
data= parse.parse('house_votes_84.data')
for i in range(10,300,60):
    withPruning = []
    withoutPruning = []
    x_axis.append(i)
    for j in range(100):
        random.shuffle(data)
        train = data[:i]
        valid =  data[i:6*i/5]
        test = data[6*i/5:]
        tree = ID3.ID3(train, 0)
        acc = ID3.test(tree, train)
        print "training accuracy: ", acc
        acc = ID3.test(tree, valid)
        print "validation accuracy: ", acc
        acc = ID3.test(tree, test)
        print "test accuracy: ", acc
        ID3.prune(tree, valid)
        acc = ID3.test(tree, train)
        print "pruned tree train accuracy: ", acc
        acc = ID3.test(tree, valid)
        print "pruned tree validation accuracy: ", acc
        acc = ID3.test(tree, test)
        print "pruned tree test accuracy: ", acc
        withPruning.append(acc)
        tree = ID3.ID3(train + valid, 'damocrat')
        acc = ID3.test(tree, test)
        print "no pruning test accuracy: ", acc
        withoutPruning.append(acc)
    without_prune.append(sum(withoutPruning)/len(withoutPruning))
    with_prune.append(sum(withPruning)/len(withPruning))

import matplotlib.pyplot as plt

sub_axix = filter(lambda x:x%200==0, x_axis)
plt.title('Result Analysis')
plt.plot(x_axis,with_prune,color='green',label='with pruning')
plt.plot(x_axis,without_prune,color='red',label= 'without pruning')
plt.legend()
plt.xlabel('traning data size')
plt.ylabel('acc')
plt.savefig('temp.png')
plt.show()