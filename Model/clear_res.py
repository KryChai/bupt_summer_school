import os

# 指定要搜索的目录路径
directory_path = '/tmp/U_Net/XCAD/test/res/'

# 定义文件模式
file_pattern = '_res_res100epoch'

# 删除所有包含指定模式的文件
for filename in os.listdir(directory_path):
    if file_pattern in filename:
        file_path = os.path.join(directory_path, filename)
        try:
            os.remove(file_path)
            print(f'删除文件: {file_path}')
        except Exception as e:
            print(f'无法删除文件: {file_path}, 原因: {e}')
