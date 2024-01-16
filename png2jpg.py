# from PIL import Image
# import os
#
# # 获取当前目录下的所有文件名
# file_list = os.listdir("./rice-seg-voc/JPEGImages")
#
# for file in file_list:
#     # 判断文件类型是否为PNG
#     if file.endswith(".png"):
#         try:
#             # 打开并读取PNG图像
#             image = Image.open(file)
#
#             # 修改文件后缀为JPG
#             new_name = file[:-4] + ".jpg"
#
#             # 保存为JPG格式
#             image.save(new_name, "JPEG")
#
#             print("已将", file, "转换为", new_name)
#
#         except Exception as e:
#             print("无法处理文件", file, ":", str(e))

import os
from PIL import Image


# 获取指定目录下的所有png图片
def get_all_png_files(dir):
    files_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                files_list.append(os.path.join(root, file))
    return files_list


# 批量转换png图片为jpg格式
def png2jpg(files_list):
    for file in files_list:
        img = Image.open(file)
        new_file = os.path.splitext(file)[0] + '.jpg'
        img.convert('RGB').save(new_file)


if __name__ == '__main__':
    dir = './rice-seg-voc/JPEGImages'  # png图片目录
    files_list = get_all_png_files(dir)
    png2jpg(files_list)
