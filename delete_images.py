import os

initial = 25
count =0
while count<initial:
    count = count+1
    os.remove(f'forwardFace{count}.jpg')
    os.remove(f'leftFace{count}.jpg')
    os.remove(f'rightFace{count}.jpg')
    os.remove(f'upFace{count}.jpg')
    os.remove(f'downFace{count}.jpg')
print("Files removed successfully!")

