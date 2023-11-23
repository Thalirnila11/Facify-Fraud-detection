nums=[1,2,3,3,3,4,5]
replace=1
count=0
for i in range(1,len(nums)):
    if nums[i] == nums[i-1]:
        count+=1
        replace+=1
    if count>=2:
        nums[replace]=nums[i]
        replace+=1
        
del nums[replace:]

print(nums)
print(len(nums))