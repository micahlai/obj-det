import os

home_dir = '/lab/micah/obj-det/testing runs'

count = 0
# for root,dirs,files in os.walk(home_dir):
#     if os.path.basename(root) == 'models':
#         for dir in dirs:
#             print(dir)
#             count += 1

for root,dirs,files in os.walk(home_dir):
    for f in files:
        if(f=='args.yaml'):
            print(root)
            count += 1

print(count)