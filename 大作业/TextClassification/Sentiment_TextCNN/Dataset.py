import os


def get_file_list(source_dir):
    file_list = []  # 文件路径名列表
    # os.walk()遍历给定目录下的所有子目录，每个walk是三元组(root,dirs,files)
    # root 所指的是当前正在遍历的这个文件夹的本身的地址
    # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    # 遍历所有评论
    for root, dirs, files in os.walk(source_dir):
        file = [os.path.join(root, filename) for filename in files]
        file_list.extend(file)
    return file_list


def get_label_list(file_list):
    # 提取出标签名
    label_name_list = [file.split("\\")[4] for file in file_list]
    # 标签名对应的数字
    label_list = []
    for label_name in label_name_list:
        if label_name == "neg":
            label_list.append(0)
        elif label_name == "pos":
            label_list.append(1)
    return label_list
