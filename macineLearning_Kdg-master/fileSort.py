import os

nums={}

dir='/project/python_project/carDetect/project/test_set/test_1'
for f_dir in os.listdir(dir):
    fileNum=list(f_dir.split('.'))[0]
    fileNum = int(fileNum)
    if not nums.get(fileNum):
        nums[fileNum]=1
    else:
        nums[fileNum] += 1
sorted_dict=sorted(nums.items(), key = lambda item: item[0], reverse=False)
print(sorted_dict)