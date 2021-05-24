import numpy as np

# read dataset
#train_full

x_full=np.loadtxt("train_full.txt", delimiter=',',dtype= str)
x_sub=np.loadtxt("train_sub.txt", delimiter=',',dtype= str)
x_noisy=np.loadtxt("train_noisy.txt", delimiter=',',dtype= str)
# Your function/class method should return:
# 1. a NumPy array of shape (N,K) representing N training instances of K attributes;
# 2. a NumPy array of shape (N, ) containing the class label for each N instance. The class label
# should be a string representing the character, e.g. "A", "E".

#How many samples/instances are there?
print(x_full.shape)
print(x_sub.shape)
print(x_noisy.shape)

#split dataset
y_full=x_full[:,16]
y_sub=x_sub[:,16]
y_noisy=x_noisy[:,16]


x_full=x_full[:,0:16].astype(int)
x_sub=x_sub[:,0:16].astype(int)
x_noisy=x_noisy[:,0:16].astype(int)


#How many unique class labels (characters to be recognised) are there?
#they all have O,C,Q,G,A,E
y_full_class=set(y_full)
y_sub_class=set(y_sub)
y_noise_class= set(y_noisy)

print(y_full_class)
print(y_sub_class)
print(y_noise_class)


#What is the distribution across the classes (e.g. 40% ‘A’s, 20% ‘C’s)?
print("+++++++++++++train_full class distribution+++++++++++++++")
y_full_A=len(y_full[y_full=='A'])/len(y_full)
y_full_C=len(y_full[y_full=='C'])/len(y_full)
y_full_G=len(y_full[y_full=='G'])/len(y_full)
y_full_E=len(y_full[y_full=='E'])/len(y_full)
y_full_O=len(y_full[y_full=='O'])/len(y_full)
y_full_Q=len(y_full[y_full=='Q'])/len(y_full)

print(y_full_A)
print(y_full_C)
print(y_full_G)
print(y_full_E)
print(y_full_O)
print(y_full_Q)

print("+++++++++++++train_sub class distribution+++++++++++++++")
y_sub_A=len(y_sub[y_sub=='A'])/len(y_sub)
y_sub_C=len(y_sub[y_sub=='C'])/len(y_sub)
y_sub_G=len(y_sub[y_sub=='G'])/len(y_sub)
y_sub_E=len(y_sub[y_sub=='E'])/len(y_sub)
y_sub_O=len(y_sub[y_sub=='O'])/len(y_sub)
y_sub_Q=len(y_sub[y_sub=='Q'])/len(y_sub)

print(y_sub_A)
print(y_sub_C)
print(y_sub_G)
print(y_sub_E)
print(y_sub_O)
print(y_sub_Q)

print("+++++++++++++train_noisy class distribution+++++++++++++++")
y_noisy_A=len(y_noisy[y_noisy=='A'])/len(y_noisy)
y_noisy_C=len(y_noisy[y_noisy=='C'])/len(y_noisy)
y_noisy_G=len(y_noisy[y_noisy=='G'])/len(y_noisy)
y_noisy_E=len(y_noisy[y_noisy=='E'])/len(y_noisy)
y_noisy_O=len(y_noisy[y_noisy=='O'])/len(y_noisy)
y_noisy_Q=len(y_noisy[y_noisy=='Q'])/len(y_noisy)

print(y_noisy_A)
print(y_noisy_C)
print(y_noisy_G)
print(y_noisy_E)
print(y_noisy_O)
print(y_noisy_Q)


#Are the samples balanced across all the classes, or are they biased towards one or two classes?
#in train_full and train_noisy the classes are banlanced, all classes 17%, but in train_sub they are not balanced, cs are two times than it should be
#G and Q have little porportion, E have a  slightly higher porportion
#mean and range of the attribute of the train_full and train_sub is almost the same, with a slighly change
attribute_full_range=x_full.max(axis=0)-x_full.min(axis=0)
print("Attributes range in train_full",attribute_full_range)
attribute_sub_range=x_sub.max(axis=0)-x_sub.min(axis=0)
print("Attributes range in train_sub",attribute_sub_range)
attribute_full_mean=x_full.mean(axis=0)
print("Attributes mean in train_full",attribute_full_mean)
attribute_sub_mean=x_sub.mean(axis=0)
print("Attributes mean in train_sub",attribute_sub_mean)


