import xlrd #需要1.2.0版本的，2.0以上的版本只能读取.xls类型的文件
import csv

# 读取文件(.xlsx .xls .csv) 然后返回字典数据
def readFile(filePath):
    try:
        fileType = filePath.split(".")[-1]
        print(f'{filePath}\t{fileType}')

        if fileType == 'xlsx' or fileType=='xls':
            res = []
            wb = xlrd.open_workbook(filePath)
            sh = wb.sheet_by_index(0)
            sheet_1 = wb.sheet_by_name("Sheet1")
            title = []
            # title: 列名list
            for item in sh.row_values(0):
                title.append(item)
# 处理excel
            fz_i = []
            fz_v = []
            nb_i = []
            nb_v = []
            sr_i = []
            sr_v = []
            # 时间列作为x轴
            times = []
            # 只取20个点
            rows = int(sheet_1.nrows/20)
            for i in range(1,sheet_1.nrows,rows):
                times.append(sheet_1.row_values(i)[0])
                for j in range(0,sheet_1.ncols):
                    # if title[j] == '负载电流':
                    #     fz_i.append([title[j],sheet_1.row_values(i)[0],sheet_1.row_values(i)[j]])
                    # elif title[j] == '负载电压':
                    #     fz_v.append([title[j],sheet_1.row_values(i)[0],sheet_1.row_values(i)[j]])
                    # elif title[j] == '逆变电流':
                    #     nb_i.append([title[j],sheet_1.row_values(i)[0],sheet_1.row_values(i)[j]])
                    # elif title[j] == '逆变电压':
                    #     nb_v.append([title[j],sheet_1.row_values(i)[0],sheet_1.row_values(i)[j]])
                    # elif title[j] == '输入电流':
                    #     sr_i.append([title[j],sheet_1.row_values(i)[0],sheet_1.row_values(i)[j]])
                    # elif title[j] == '输入电压':
                    #     sr_v.append([title[j],sheet_1.row_values(i)[0],sheet_1.row_values(i)[j]])
                    if title[j] == '负载电流':
                        fz_i.append(sheet_1.row_values(i)[j])
                    elif title[j] == '负载电压':
                        fz_v.append(sheet_1.row_values(i)[j])
                    elif title[j] == '逆变电流':
                        nb_i.append(sheet_1.row_values(i)[j])
                    elif title[j] == '逆变电压':
                        nb_v.append(sheet_1.row_values(i)[j])
                    elif title[j] == '输入电流':
                        sr_i.append(sheet_1.row_values(i)[j])
                    elif title[j] == '输入电压':
                        sr_v.append(sheet_1.row_values(i)[j])
            # list_sum = fz_i+fz_v+nb_i+nb_v+sr_i+sr_v
            # list_sum.insert(0,['type','time','value'])
            list_sum={
                'fz_i':fz_i,
                'fz_v':fz_v,
                'nb_i':nb_i,
                'nb_v':nb_v,
                'sr_i':sr_i,
                'sr_v':sr_v,
                'times':times
            }
# /处理excel
           # data = []
            # 实现第一行为key，剩下的为value 转为字典了
           # [[data.append({title[index]: transfer(sh.row_values(it)[index]) for index in range(0,len(title))})] for it in range(1,sh.nrows)]
            return list_sum
        elif fileType == "csv":
            data = []
            with open(filePath) as csvfile:
                rows = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
                title = next(rows)  # 读取第一行每一列的标题
                [[data.append({title[index]: transfer(it[index]) for index in range(0, len(title))})] for it in rows]
            return data
        else:
            return -1
    except(EOFError):
        print("转化过程出错！")
        print(EOFError)
        return -1


# 字符串输入，转成相应的类型
def transfer(string):
    try:
        if float(string) == float(int(float(string))):
            return int(string)
        else:
            return float(string)
    except:
        pass
    return True if string.lower() == 'true' else (False if string.lower() == 'false' else string)

