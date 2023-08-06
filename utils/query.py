import datetime
from datetime import date, datetime,timedelta,time
from pymysql import *
import xlrd
import json
conn = connect(host='localhost', user='root',password='123456', database='chz1',port=3306)
cursor = conn.cursor()

def querys(sql,params,type='no_select'):
    params = tuple(params)
    #重新连接数据库，以防连接超时
    conn.ping(reconnect=True)
    cursor.execute(sql,params)
    if type != 'no_select':
        data_list = cursor.fetchall()
        conn.commit()
        return data_list
    else:
        conn.commit()
        return '数据库语句执行成功'
def select_result2(uname):
    '''
    t:在t表中找uname用户最近的记录
    tr2中
    return 每个label的出现次数
    '''
    sql = '''select tname from test_result2 where time = (select max(time) t from test_result2 where id=\'%s\')'''%(uname)
    conn.ping(reconnect=True)
    cursor.execute(sql)
    tname = cursor.fetchall()
    if len(tname)==0:
        return 'no_history'
    tname = tname[0]
    sql2 = ''' select * from %s'''%tname
    conn.ping(reconnect=True)
    cursor.execute(sql2)
    labels = cursor.fetchall()
    print(labels)
    list = [0,0,0,0]
    for i in range(len(labels)):
        if labels[i][0] ==1:
            list[0] =list[0]+1
        elif labels[i][0]==2:
            list[1]=list[1]+1
        elif labels[i][0]==3:
            list[2]=list[2]+1
        elif labels[i][0]==4:
            list[3]=list[3]+1
    print(list)
    return list
def select_last_info(uname):
    #准备
    sql = '''select tname,model,time from test_result2 where time = (select max(time) t from test_result2 where id=\'%s\')'''%(uname)
    conn.ping(reconnect=True)
    cursor.execute(sql)
    re = cursor.fetchall()
    tname = str(re[0][0])
    model = re[0][1]
    up_time = str(re[0][2])
    main_name = tname.split('_',1)
    main_name=main_name[1]
    print(main_name)
    # # 用户输入结果名称
    # sql3 = '''select input_name from result_name_map where main_name=\'%s\' ''' % (main_name)
    # conn.ping(reconnect=True)
    # cursor.execute(sql3)
    # input_name = cursor.fetchall()
    # print(input_name)
    # input_name = input_name[0]
    return model,up_time
def excel_input(name,path):
    '''
    :param name: 前缀_excel名字
    :param path:excel存储位置
    excel结构：times，6 attribues
    '''
    # 打开数据所在的工作簿，以及选择存有数据的工作表
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)  # 返回第一个表的对象
    sql_drop = 'drop table if exists %s'%name
    conn.ping(reconnect=True)
    cursor.execute(sql_drop)
    sql_create ='''CREATE TABLE if not exists %s 
                (times time ,
                fz_i float,
                fz_v float,
                nb_i float,
                nb_v float,
                sr_i float,
                sr_v float)'''%name
    conn.ping(reconnect=True)
    cursor.execute(sql_create)
    # 创建一个for循环迭代读取xls文件每行数据的, 从第二行开始是要跳过标题行
    for r in range(1, sheet.nrows):
        time = sheet.cell(r, 0).value
        # print(time)
        # print("type",type(time))
        # time = datetime.strptime(time,'%H:%M:%S')
        fz_i = sheet.cell(r, 1).value
        fz_v = sheet.cell(r, 2).value
        nb_i = sheet.cell(r, 3).value
        nb_v = sheet.cell(r, 4).value
        sr_i = sheet.cell(r, 5).value
        sr_v = sheet.cell(r, 6).value
        #print(type(time))
        values = (name.split('.')[0],time,fz_i, fz_v, nb_i, nb_v, sr_i, sr_v)
        # 执行sql语句,传字符串的时候一定要加\'%s\'，两边单引号
        conn.ping(reconnect=True)
        cursor.execute('insert into %s values (\'%s\', %f, %f, %f, %f, %f, %f)' %(name,time,fz_i, fz_v, nb_i, nb_v, sr_i, sr_v))
        #cursor.execute(sql_insert, values)
    conn.commit()
    columns = str(sheet.ncols)
    rows = str(sheet.nrows)
    print("导入 " + columns + " 列 " + rows + " 行数据到MySQL数据库!")
def result_input(name,path,col):
    '''
    excel结构: n*4列proba or n*1列 labels
    :param name: 前缀（tr_）+excel name
    :param path: result存储相对路径
    :param col:
    :return:
    '''
    # 打开数据所在的工作簿，以及选择存有数据的工作表
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)  # 返回第一个表的对象
    sql_drop = 'drop table if exists %s'%name
    cursor.execute(sql_drop)
    if col == 4:
        sql_create ='''CREATE TABLE if not exists %s 
                    (normal float,
                    a_shadow float,
                    p_shadow float,
                    open_c float)'''%(name)
        conn.ping(reconnect=True)
        cursor.execute(sql_create)
        # 创建一个for循环迭代读取xls文件每行数据的, 从第二行开始是要跳过标题行
        for r in range(1, sheet.nrows):
            normal = sheet.cell(r, 0).value
            # print(type(time))
            # time = datetime.datetime.strptime(time,'%H:%M:%S')
            a_shadow = sheet.cell(r, 1).value
            p_shadow = sheet.cell(r, 2).value
            open_c = sheet.cell(r, 3).value
            #print(type(time))
            values = (name.split('.')[0],normal,a_shadow,p_shadow,open_c)
            # 执行sql语句,传字符串的时候一定要加\'%s\'，两边单引号
            conn.ping(reconnect=True)
            cursor.execute('insert into %s values (%f, %f, %f,%f)' %(name,normal,a_shadow,p_shadow,open_c))
            #cursor.execute(sql_insert, values)
    elif col ==1 :
        sql_create = '''CREATE TABLE if not exists %s 
                            (types float)''' % (name)
        conn.ping(reconnect=True)
        cursor.execute(sql_create)
        # 创建一个for循环迭代读取xls文件每行数据的, 从第二行开始是要跳过标题行
        for r in range(1, sheet.nrows):
            types = sheet.cell(r, 0).value
            values = (name.split('.')[0], types)
            # 执行sql语句,传字符串的时候一定要加\'%s\'，两边单引号
            conn.ping(reconnect=True)
            cursor.execute('insert into %s values (\'%s\')' % (name.split('.')[0], types))
    conn.commit()
    columns = str(sheet.ncols)
    rows = str(sheet.nrows)
    print("导入 " + columns + " 列 " + rows + " 行数据到MySQL数据库!")

# 向record传入完整名字
# xname在result插入时是这个result对应的test_x数据集名称，在map插入时是用户给result取的输入的名称
def record_insert(uname,tname,type,x_name,model):
    today_time = str(datetime.now()).split('.')[0]
    today_time = datetime.strptime(today_time, '%Y-%m-%d %H:%M:%S')
    if type == 'vis':
        sql_ins = '''INSERT INTO vis_dir (tname,id,time)
                    VALUES(\'%s\',\'%s\',\'%s\');'''%(tname.split('.')[0],uname,today_time)
    elif type == 'test':
        sql_ins = '''INSERT INTO test_dir (tname,id,time,model)
                    VALUES(\'%s\',\'%s\',\'%s\',\'%s\');'''%(tname.split('.')[0],uname,today_time,model)
    elif type == 'test_result':
        sql_ins = '''INSERT INTO test_result (tname,id,test_tname,time,model)
                            VALUES(\'%s\',\'%s\',\'%s\',\'%s\',\'%s\');''' % (tname.split('.')[0], uname, x_name,today_time,model)
    elif type == 'test_result2':
        sql_ins = '''INSERT INTO test_result2 (tname,id,test_tname,time,model)
                            VALUES(\'%s\',\'%s\',\'%s\',\'%s\',\'%s\');''' % (tname.split('.')[0], uname, x_name,today_time,model)
    elif type =='result_name_map':
        sql_ins = '''INSERT INTO result_name_map (main_name,input_name,uname,model)
                            VALUES(\'%s\',\'%s\',\'%s\',\'%s\')'''%(tname,x_name,uname,model)
    conn.ping(reconnect=True)
    cursor.execute(sql_ins)
    conn.commit()
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, timedelta):
            return obj.__str__()
        else:
            return json.JSONEncoder.default(self, obj)
def emysql_visualize(table_name):
    list_sql = ' SELECT * FROM  %s  '%table_name
    conn.ping(reconnect=True)
    cursor.execute(list_sql)  # 执行单条sql语句
    list_result = cursor.fetchall()  # 接收全部的返回结果行
    # cursor.execute(fz_i)
    # fz_iresult = cursor.fetchall()
    # cursor.execute(fz_v)
    # fz_vresult = cursor.fetchall()
    # cursor.execute(nb_i)
    # nb_iresult = cursor.fetchall()
    # cursor.execute(nb_v)
    # nb_vresult = cursor.fetchall()
    # cursor.execute(sr_i)
    # sr_iresult = cursor.fetchall()
    # cursor.execute(sr_v)
    # sr_vresult = cursor.fetchall()
    times = []
    fz_is = []
    fz_vs = []
    nb_is = []
    nb_vs = []
    sr_is = []
    sr_vs = []
    #展示平均分布的20条数据
    x = int(len(list_result)/20)
    i = 1
    for data in list_result:
        if i== x:
            times.append(str(data[0]))
            fz_is.append(data[1])
            fz_vs.append(data[2])
            nb_is.append(data[3])
            nb_vs.append(data[4])
            sr_is.append(data[5])
            sr_vs.append(data[6])
            i = 1
        else:
            i=i+1
    print(times)
    # jsonData = {}
    # jsonData['times']= times
    # jsonData['fz_is'] = fz_is
    # j = json.dumps(jsonData,cls=ComplexEncoder)
    return(times,fz_is,fz_vs,nb_is,nb_vs,sr_is,sr_vs)

def result_mysql_visualize(test_t, tr2_t):
    # 时间+6维数据分4种类，传这样4*7数组分开
    #7维数据
    drop_sql1 = 'drop table if exists temp_7'
    drop_sql2 = 'drop table if exists temp_re2'
    cursor.execute(drop_sql2)
    cursor.execute(drop_sql1)
    temp_sql1 = ' create table temp_7 as select ROW_NUMBER()over() rn_f,times, fz_i,fz_v,nb_i,nb_v,sr_i,sr_v from %s '%test_t
    # 打标1 col
    temp_sql2 = 'create table temp_re2 as select ROW_NUMBER()over() rn_f,types from %s'%tr2_t
    conn.ping(reconnect=True)
    cursor.execute(temp_sql1)  # 执行单条sql语句
    cursor.execute(temp_sql2)  # 执行单条sql语句
    list_result=[]  #4*num*7
    sele_sql = '''select fz_i,fz_v,nb_i,nb_v,sr_i,sr_v,
                    case(types)
                        when 1 then '正常'
                        when 2 then '全部遮挡'
                        when 3 then '部分遮挡'
                        when 4 then '断路'
                        else 'NAN'
                    end as types    
                    from temp_re2 a
                    inner join temp_7 b
                    on 	a.rn_f = b.rn_f'''
    cursor.execute(sele_sql)  # 执行单条sql语句
    rawData=list(cursor.fetchall()) # 接收全部的返回结果行
    # type_t =[]
    # for data in type:
    #     data_t =[data[0],data[1],data[2],data[3],data[4],data[5],data[6]]
    #     type_t.append(data_t)
    # list_result.append(type_t)
        # 0 正常，1全遮，2半遮挡，3断路
    cursor.execute(drop_sql2)
    cursor.execute(drop_sql1)
    return(rawData)

def result_mysql_visualize_dk(test_t, tr2_t, window):
    # 返回（特征+label）*个数
    #7维数据
    temp_sql1 = ' create table temp_7 as select ROW_NUMBER()over() rn_f,times,fz_i,fz_v,nb_i,nb_v,sr_i,sr_v from %s '%test_t
    # 打标1 col
    temp_sql2 = 'create table temp_re2 as select ROW_NUMBER()over() rn_f,types from %s'%tr2_t
    conn.ping(reconnect=True)
    cursor.execute(temp_sql1)  # 执行单条sql语句
    cursor.execute(temp_sql2)  # 执行单条sql语句
    sele_sql = '''select fz_i,fz_v,nb_i,nb_v,sr_i,sr_v,
                    case(types)
                        when 1 then '正常'
                        when 2 then '全部遮挡'
                        when 3 then '部分遮挡'
                        when 4 then '断路'
                        else 'NAN'
                    end as types    
                    from temp_re2 a
                    inner join temp_7 b
                    on 	a.rn_f*%f <= b.rn_f
                    and b.rn_f-%f<a.rn_f*%f'''%(window,window,window)
    conn.ping(reconnect=True)
    cursor.execute(sele_sql)  # 执行单条sql语句
    rawData =list(cursor.fetchall()) # 接收全部的返回结果行
    drop_sql1 = 'drop table if exists temp_7'
    drop_sql2 = 'drop table if exists temp_re2'
    cursor.execute(drop_sql2)
    cursor.execute(drop_sql1)
    return(rawData)
def user_info(name):
    sql = 'select email,password,identity from user where name =\'%s\' '%name
    cursor.execute(sql)
    list_result = cursor.fetchall()  # 接收全部的返回结果行
    print(list_result)
    return(list_result[0][0],list_result[0][1],list_result[0][2])
def change_pass(name,new_pass):
    sql = 'update user set password=\'%s\' where name=\'%s\' '%(new_pass,name)
    cursor.execute(sql)
    return('OK')



