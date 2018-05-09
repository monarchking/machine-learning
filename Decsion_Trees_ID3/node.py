#_*_coding:utf-8_*_
import operator
class Node:
  def __init__(self,tree):
    self.label = tree
    self.children = {}
    self.tree = tree
    self.assign()
	# you may want to add additional fields here...
  def assign(self):
    self.label = list(self.tree.keys())[0]
    self.children = self.tree[self.label]










    
