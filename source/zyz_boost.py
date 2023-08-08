import warnings
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 导入绘图包

warnings.filterwarnings("ignore")
import re
import pandas as pd
import csv
from tqdm import tqdm

little_freq = [300000, 403200, 499200, 595200, 691200, 806400, 902400, 998400, 1094400, 1209600, 1305600, 1401600,
               1497600, 1612800, 1708800, 1804800]
big_freq = [710400, 844800, 960000, 1075200, 1209600, 1324800, 1440000, 1555200, 1670400, 1766400, 1881600, 1996800,
            2112000, 2227200, 2342400, 2419200]
s_big_freq = [844800, 960000, 1075200, 1190400, 1305600, 1420800, 1555200, 1670400, 1785600, 1900800, 2035200, 2150400,
              2265600, 2380800, 2496000, 2592000, 2688000, 2764800, 2841600]


# 通用画图代码
def plt_boost_policy(data, cluster):
    data = np.array(data)

    # Calculate the sum of each row to normalize the data to percentage
    row_sums = data.sum(axis=1)
    data_percent = (data.T / row_sums * 100).T

    # Create the stacked bar chart
    plt.figure(figsize=(14, 6))
    if cluster == 'small' or 'cpu_small':
        x_raw = little_freq
    if cluster == 'big' or 'cpu_big':
        x_raw = big_freq
    if cluster == 'super':
        x_raw = s_big_freq

    rows = [x_raw[i] for i in range(len(data_percent))]

    x = np.arange(len(rows))

    bottoms = [0] * len(rows)
    col_label = ['rtg_boost', 'hiload + nomig', 'hiload + nl', 'predict load', 'normal util', 'target load',
                 'driver boost', 'no count']
    if cluster == 'cpu_small':
        col_label = ['cpu0', 'cpu1', 'cpu2', 'cpu3']
    if cluster == 'cpu_big':
        col_label = ['cpu4', 'cpu5', 'cpu6']
    for i in range(len(data_percent[0])):
        bars = plt.bar(x, data_percent[:, i], bottom=bottoms, label=col_label[i])
        bottoms = [sum(values) for values in zip(bottoms, data_percent[:, i])]
        # Add percentage labels to each bar
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2,
                         f"{height:.1f}%", ha='center', va='center', fontsize=8)

    plt.xticks(x, rows)
    plt.xlabel('freq point')
    plt.ylabel('Percentage')
    plt.title('{} Cluster boost policy'.format(cluster))
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.grid(axis='y')
    plt.tight_layout()

    plt.show()


# 过滤掉列表中的假值（False、None、0、空字符串等），只保留真值。
def filter_false(lst):
    return list(filter(bool, lst))


# 检查数字num的千位、百位、十位和个位是否都等于1
def check_digits_for_1(num):
    thousand_digit = num // 1000
    hundred_digit = (num // 100) % 10
    ten_digit = (num // 10) % 10
    unit_digit = num % 10
    return thousand_digit == 1, hundred_digit == 1, ten_digit == 1, unit_digit == 1


# 统计列表中值的计数，并排序
def count_and_sort(lst):
    # 使用Counter类统计列表中每个值的计数
    counter = Counter(lst)
    # 按计数值进行排序，得到一个列表，元素为元组(值, 计数)
    sorted_counts = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts


# 抽取trace数据
def analysis_freq_trace(trace_path):
    print(f'transferring {trace_path}')

    # 打开trace文件解析数据
    trace_file = open(trace_path, encoding="utf-8")
    '''
           <...>-12510 [007] d.ha   209.474831: get_my_util: cpu 7 519
           <...>-12510 [007] d.ha   209.474831: get_my_util: rtg_boost 7 519
           <...>-12510 [007] d.ha   209.474833: get_my_util: nl 7 1024
           <...>-12510 [007] d.ha   209.474834: get_my_util: max 7 1024
    '''
    pattern_freq = re.compile(
        rf'\d+.\d+: get_my_util: .* \d \d+')

    lines = trace_file.readlines()

    # 创建缓存数组
    time_list = []
    cpu_list = []
    before_list = []
    rtg_list = []
    hs_list = []
    nl_list = []
    pl_list = []
    after_list = []

    flags = ['cpu', 'rtg_boost', 'hispeed_util', 'nl', 'pl']
    ptr = 0  # 标签指针
    len_after = 0
    for line in tqdm(lines, desc='解析util中间值', position=0):
        my_util = pattern_freq.findall(line)

        if my_util != []:
            # 处理文本数据转为指定类型
            s = my_util[0].split(' ')  # ['209.192590:', 'get_my_util:', 'cpu', '0', '322']
            time_stamp = float(s[0].split(':')[0])
            flag = s[2]
            cpu = int(s[3])
            value = int(s[4])

            # 处理残缺值，补0，max是最终util，多核只出现一次，故加倍存储
            if flag == 'max':
                after_list += [value] * (len(before_list) - len(after_list))
                # if cpu == 0:
                #     after_list += [value]*(len(before_list)-len(after_list))
                # elif cpu == 4:
                #     after_list += [value]*3
                # elif cpu == 7:
                #     after_list += [value]
            else:
                if flag == 'cpu':
                    time_list.append(time_stamp)
                    cpu_list.append(cpu)

                while flag != flags[ptr]:
                    if ptr == 0:
                        before_list.append(0)
                    elif ptr == 1:
                        rtg_list.append(0)
                    elif ptr == 2:
                        hs_list.append(0)
                    elif ptr == 3:
                        nl_list.append(0)
                    elif ptr == 4:
                        pl_list.append(0)
                    ptr = (ptr + 1) % 5

                if flag == flags[ptr]:
                    if ptr == 0:
                        before_list.append(value)
                    elif ptr == 1:
                        rtg_list.append(value)
                    elif ptr == 2:
                        hs_list.append(value)
                    elif ptr == 3:
                        nl_list.append(value)
                    elif ptr == 4:
                        pl_list.append(value)
                    ptr = (ptr + 1) % 5

    print(time_list[-10:])
    print(cpu_list[-10:])
    print(before_list[-10:])
    print(after_list[-10:])
    print(1)
    # 将处理好的数据转为csv
    df = pd.DataFrame(
        {"timestamp": time_list, "cpu": cpu_list, "before": before_list, "rtg": rtg_list, "hs": hs_list,
         "nl": nl_list, "pl": pl_list, "after": after_list})

    save_path = r"./freq_details.csv"
    df.to_csv(save_path, sep=',', index=False, header=True)
    print("数据处理完毕，另存为csv文件")
    trace_file.close()


# 分析csv文件
def analysis_freq_contribute(read_csv, write_csv):
    flags = ['no_boost', 'rtg_boost', 'hispeed_util', 'nl', 'pl', 'master']  # CPU影响占比，哪个CPU决定最终频
    cpus = ['CPU0', 'CPU1', 'CPU2', 'CPU3', 'CPU4', 'CPU5', 'CPU6', 'CPU7']
    data0 = np.zeros((8, 6))
    contribute = pd.DataFrame(data=data0, index=cpus, columns=flags)

    data = pd.read_csv(read_csv, header=0)
    data = pd.DataFrame(data)
    # 遍历 CSV 文件的每一行
    for index, row in data.iterrows():  # ('timestamp', 209.19259) ('cpu', 0.0) ('before', 322.0) ('rtg', 322.0) ('hs', 322.0) ('nl', 0.0) ('pl', 322.0) ('after', 322.0)
        if row['after'] == row['pl']:
            contribute.iloc[int(row['cpu']), 5] += 1

        if row['after'] == row['before']:
            contribute.iloc[int(row['cpu']), 0] += 1
        elif row['after'] == row['rtg']:
            contribute.iloc[int(row['cpu']), 1] += 1
        elif row['after'] == row['hs']:
            contribute.iloc[int(row['cpu']), 2] += 1
        elif row['after'] == row['nl']:
            contribute.iloc[int(row['cpu']), 3] += 1
        elif row['after'] == row['pl']:
            contribute.iloc[int(row['cpu']), 4] += 1

    contribute.to_csv(write_csv, sep=',', index=True, header=True)
    print("各指标贡献情况分析完毕，已生成csv文件...")


#  画饼图，这里画每个簇是由哪个CPU最终决定util
def plot_pie(ax, data, title, lable, bbox_to_anchor=(0.06, 1), dpi=100):  # x:横轴子图数，y:纵轴子图数
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
    plt.rcParams["font.size"] = 12
    plt.rcParams['font.family'] = 'Times New Roman'
    font1 = {'family': 'SimHei',
             'size': 16,
             }
    ax.set_title(title, font1)
    ax.pie(data, labels=lable, autopct='%3.1f%%')
    ax.legend(lable, loc="upper center", bbox_to_anchor=bbox_to_anchor, fontsize='small',
              handlelength=1.8, borderpad=0.1, frameon=False)


#  画百分比堆积图，这里看各指标的影响，包括CPU和指标两个考虑维度
def percentage_bar(df, mycolor='GnBu'):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
    plt.rcParams["font.size"] = 15
    plt.rcParams['font.family'] = 'Times New Roman'
    font1 = {'family': 'SimHei',
             'size': 16,
             }

    labels = df.index.tolist()  # 提取分类显示标签
    results = df.to_dict(orient='list')  # 将数值结果转化为字典
    category_names = list(results.keys())  # 提取字典里面的类别（键-key）
    data = np.array(list(results.values()))  # 提取字典里面的数值（值-value）

    category_colors = plt.get_cmap(mycolor)(np.linspace(0.1, 0.9, data.shape[0]))
    # 设置占比显示的颜色，可以自定义，修改括号里面的参数即可，如下
    # category_colors = plt.get_cmap('hot')(np.linspace(0.1, 0.9, data.shape[0]))

    fig, ax = plt.subplots(figsize=(12, 5))  # 创建画布，开始绘图
    ax.invert_yaxis()  # 这个可以通过设置df中columns的顺序调整
    ax.set_yticklabels(labels=labels)  # 显示x轴标签，并旋转90度
    ax.set_xlim(0, 1)  # 设置x轴的显示范围
    starts = 0  # 绘制基准
    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        heights = data[i, :] / data.sum(axis=0)  # 计算出每次遍历时候的百分比
        ax.barh(labels, heights, left=starts, height=0.8, label=colname, color=color, edgecolor='gray')  # 绘制柱状图
        xcenters = starts + heights / 2  # 进行文本标记位置的选定
        starts += heights  # 核心一步，就是基于基准上的百分比累加
        # print(starts) 这个变量就是能否百分比显示的关键，可以打印输出看一下
        percentage_text = data[i, :] / data.sum(axis=0)  # 文本标记的数据

        r, g, b, _ = color  # 这里进行像素的分割
        # text_color = 'white' if r * g * b < 0.5 else 'k'  # 根据颜色基调分配文本标记的颜色
        text_color = 'black'
        for y, (x, c) in enumerate(zip(xcenters, percentage_text)):
            if c*100 > 5:  # 小于10的不显示
                ax.text(x, y, f'{round(c * 100, 2)}%', ha='center', va='center',
                        color=text_color)  # 添加文本标记
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0.5, 1),
              handlelength=1.6, borderpad=0.5, loc='lower center', fontsize='medium')  # 设置图例
    return fig, ax  # 返回图像


def plot_freq_contribute(csv_path):
    data = pd.read_csv(csv_path, header=0, index_col=0)
    data = pd.DataFrame(data)

    # fixme 第一幅，CPU最终频点的决定核心
    fig = plt.figure(figsize=(15, 5), dpi=100)
    fig.subplots_adjust(left=0.07, bottom=0.1, right=0.96, top=0.9, wspace=0.2, hspace=0.2)  # 调整子图间距按百分比

    label = ['CPU0', 'CPU1', 'CPU2', 'CPU3']
    title = '小核簇各核心贡献率'
    data0 = data.iloc[0:4, -1].values
    ax = fig.add_subplot(131)
    plot_pie(ax, data0, title, label, bbox_to_anchor=(0.06, 1))

    label = ['CPU4', 'CPU5', 'CPU6']
    title = '大核簇各核心贡献率'
    data4 = data.iloc[4:7, -1].values
    ax = fig.add_subplot(132)
    plot_pie(ax, data4, title, label, bbox_to_anchor=(0.06, 1))

    label = ['CPU0', 'CPU1', 'CPU2', 'CPU3', 'CPU4', 'CPU5', 'CPU6', 'CPU7']
    title = '全部各核心贡献率'
    data7 = data.iloc[:, -1].values
    ax = fig.add_subplot(133)
    plot_pie(ax, data7, title, label, bbox_to_anchor=(0.06, 1.12))

    plt.savefig(rf"./pic/cpu_con.png", dpi=100)  # 保存图片，分辨率为100
    # plt.savefig(rf'./cpu_con.svg', format='svg', dpi=100)  # 输出
    plt.show()  # 显示图片,这个可以方便后面自动关闭

    # fixme 第二幅，各指标对util boost的影响
    percentage_bar(data.iloc[:,:-1])
    plt.xticks(rotation=0)
    plt.savefig('./pic/bar_cpu_boost.png')
    plt.show()

    clus = ['SMALL', 'BIG', 'SUPER']
    flags = ['no_boost', 'rtg_boost', 'hispeed_util', 'nl', 'pl']
    boost_clus = pd.DataFrame(data=np.zeros((3, 5)), index=clus, columns=flags)
    for x in range(len(clus)):
        for y in range(len(flags)):
            if x == 0:
                boost_clus.iloc[x, y] += data.iloc[0:4, y].sum()
            elif x == 1:
                boost_clus.iloc[x, y] += data.iloc[4:7, y].sum()
            else:
                boost_clus.iloc[x, y] += data.iloc[7, y].sum()
    percentage_bar(boost_clus)
    plt.xticks(rotation=0)
    plt.savefig('./pic/bar_cluster_boost.png')
    plt.show()

    # fixme 第三幅，各指标对util boost的影响
    percentage_bar(data.iloc[:, :-1].T, 'Reds')
    plt.xticks(rotation=0)
    plt.savefig('./pic/bar_boost_cpu.png')
    plt.show()

    percentage_bar(boost_clus.T, 'Reds')
    plt.xticks(rotation=0)
    plt.savefig('./pic/bar_boost_cluster.png')
    plt.show()


def read_boost_next_freq(trace_path):
    global freq
    import codecs

    with codecs.open(trace_path, 'rb') as file:
        Kernel_Trace_Data = pd.read_table(file, header=None, error_bad_lines=False, warn_bad_lines=False,
                                          encoding='utf-8')

    Kernel_Trace_Data_List = Kernel_Trace_Data.values.tolist()
    up = 'update_cpu_busy_time:'
    snf = 'sugov_next_freq_shared:'
    sus = 'sugov_update_single:'
    for i in range(len(Kernel_Trace_Data_List)):
        Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i][0].split("=")
        # Kernel_Trace_Data_List[i] = Kernel_Trace_Data_List[i].split(",")
    for i in range(len(Kernel_Trace_Data_List)):
        tmp = []
        tmp1 = []
        tmp2 = []
        for j in range(len(Kernel_Trace_Data_List[i])):
            if (len(Kernel_Trace_Data_List[i][j].split()) > 3):
                tmp2 = Kernel_Trace_Data_List[i][j].split()
                tmp1 = tmp2[3]
                tmp1 = tmp1[0:-1]
                tmp2[3] = tmp1
                tmp += tmp2

            else:
                tmp += Kernel_Trace_Data_List[i][j].split()
        Kernel_Trace_Data_List[i] = tmp

    Kernel_Trace_Data_List = filter_false(Kernel_Trace_Data_List)

    Kernel_Trace_Data_List = Kernel_Trace_Data_List[11:]
    k = Kernel_Trace_Data_List
    # print(k)
    df1 = Kernel_Trace_Data_List[:0]
    small_freq_list = [0] * 16
    big_freq_list = [0] * 16
    super_freq_list = [0] * 19

    small_j_util_freq_list = [0] * 16
    big_j_util_freq_list = [0] * 16
    super_j_util_freq_list = [0] * 19
    # 用于统计boost策略对req freq的贡献
    small_freq_policy_list = [[] for _ in range(16)]
    big_freq_policy_list = [[] for _ in range(16)]
    super_freq_policy_list = [[] for _ in range(19)]
    # 用于统计各cpu对最终util的贡献
    small_freq_cpu_list = [[] for _ in range(16)]
    big_freq_cpu_list = [[] for _ in range(16)]

    diff_count = 0
    total_count = 0
    small_count = 0
    small_total_count = 0
    big_count = 0
    big_total_count = 0
    super_count = 0
    super_total_count = 0
    for i in range(len(Kernel_Trace_Data_List)):
        if (k[i][4] == 'sugov_next_freqs:' or k[i][4] == 'boost_policy:'):
            df1.append(Kernel_Trace_Data_List[i])
    for x in tqdm(range(0, len(df1))):
        total_count = total_count + 1
        cpu_util = -1
        for y in range(len(df1[x])):
            if (df1[x][y] == 'cpu'):
                cpu = int(df1[x][y + 1])
            if (df1[x][y] == 'cpu_util'):
                cpu_util = int(df1[x][y + 1])
            if (df1[x][y] == 'util'):
                util = int(df1[x][y + 1])
        if (df1[x][4] == 'sugov_next_freqs:'):
            if cpu == 0:
                small_total_count = small_total_count + 1
            if cpu == 4:
                big_total_count = big_total_count + 1
            if cpu == 7:
                super_total_count = super_total_count + 1
        if cpu == 0 and x + 5 < len(df1) - 1 and int(df1[x + 4][6]) == 0 and cpu_util > 0:
            last_util = util
            cpu_util_list = []
            for j in range(4):
                j_cpu_util = int(df1[x + j][8])
                j_util = int(df1[x + j][10])
                j_policy = int(df1[x + j][12])
                if (j == 0):
                    max_policy = j_policy
                    max_idx = 0
                if (j_util > last_util):
                    max_idx = j
                    max_policy = j_policy
                last_util = j_util
                cpu_util_list.append(j_cpu_util)

            # 计算基于原始util的频率
            max_j_util = max(cpu_util_list)
            max_freq = int(df1[x + 4][16])
            raw_freq = int(max_j_util * 1.25 * 1804800 / 325)
            util_to_freq = little_freq[np.searchsorted(little_freq, raw_freq)] if little_freq[0] <= raw_freq <= \
                                                                                  little_freq[-1] \
                else (little_freq[0] if raw_freq < little_freq[0] else little_freq[-1])
            small_j_util_freq_list[little_freq.index(util_to_freq)] = small_j_util_freq_list[
                                                                          little_freq.index(util_to_freq)] + 1
            # 计算基于最终经过boost的util的频率
            raw_freq = int(last_util * 1.25 * 1804800 / 325)
            freq = little_freq[np.searchsorted(little_freq, raw_freq)] if little_freq[0] <= raw_freq <= little_freq[-1] \
                else (little_freq[0] if raw_freq < little_freq[0] else little_freq[-1])
            # small_freq_policy_list[little_freq.index(freq)].append(max_policy)
            # 在sugov统计的频率记为req_freq
            normal_freq = int(df1[x + 4][12])
            req_freq = int(df1[x + 4][14])
            # print(req_freq, freq)
            if normal_freq == req_freq:
                if (normal_freq == freq):
                    if (util_to_freq < freq):
                        small_count = small_count + 1
                        small_freq_policy_list[little_freq.index(req_freq)].append(max_policy)
                    else:
                        small_freq_policy_list[little_freq.index(req_freq)].append(0)
                        small_count = small_count + 1

                else:
                    small_freq_policy_list[little_freq.index(req_freq)].append(-1)
            else:
                small_freq_policy_list[little_freq.index(req_freq)].append(-2)
            small_freq_list[little_freq.index(req_freq)] = small_freq_list[little_freq.index(req_freq)] + 1
            # 记录这一次选频最终决定频率的cpu
            small_freq_cpu_list[little_freq.index(req_freq)].append(max_idx)

        if cpu == 4 and x + 3 <= len(df1) - 1 and int(df1[x + 3][6]) == 4 and cpu_util > 0:
            last_util = util
            cpu_util_list = []
            for j in range(3):
                j_cpu_util = int(df1[x + j][8])
                j_util = int(df1[x + j][10])
                j_policy = int(df1[x + j][12])
                if (j == 0):
                    max_policy = j_policy
                    max_idx = 0
                if (j_util > last_util):
                    max_idx = j + 0
                    max_policy = j_policy
                last_util = j_util
                cpu_util_list.append(j_cpu_util)

            # 计算基于原始util的频率
            max_j_util = max(cpu_util_list)
            max_freq = int(df1[x + 3][16])
            raw_freq = int(max_j_util * 1.25 * 2419200 / 828)
            util_to_freq = big_freq[np.searchsorted(big_freq, raw_freq)] if big_freq[0] <= raw_freq <= big_freq[
                -1] else (
                big_freq[0] if raw_freq < big_freq[0] else big_freq[-1])
            big_j_util_freq_list[big_freq.index(util_to_freq)] = big_j_util_freq_list[big_freq.index(util_to_freq)] + 1
            # 计算基于最终经过boost的util的频率
            raw_freq = int(int(df1[x + 3][8]) * 1.25 * max_freq / 828)
            freq = big_freq[np.searchsorted(big_freq, raw_freq)] if big_freq[0] <= raw_freq <= big_freq[-1] else (
                big_freq[0] if raw_freq < big_freq[0] else big_freq[-1])

            # 在sugov统计的频率记为req_freq
            normal_freq = int(df1[x + 3][12])
            req_freq = int(df1[x + 3][14])
            # print(req_freq, freq)
            if normal_freq == req_freq:
                if (normal_freq == freq):
                    if (util_to_freq < freq):
                        big_count = big_count + 1
                        big_freq_policy_list[big_freq.index(req_freq)].append(max_policy)
                    else:
                        big_count = big_count + 1
                        big_freq_policy_list[big_freq.index(req_freq)].append(0)

                else:
                    big_freq_policy_list[big_freq.index(req_freq)].append(-1)
            else:
                big_freq_policy_list[big_freq.index(req_freq)].append(-2)
            # big_total_count = big_total_count + 1
            big_freq_list[big_freq.index(req_freq)] = big_freq_list[big_freq.index(req_freq)] + 1
            # 记录这一次选频最终决定频率的cpu
            big_freq_cpu_list[big_freq.index(req_freq)].append(max_idx)

        if cpu == 7 and x + 1 <= len(df1) - 1 and int(df1[x + 1][6]) == 7 and cpu_util > 0:
            last_util = util
            cpu_util_list = []
            for j in range(2):
                j_cpu_util = int(df1[x + j][8])
                j_util = int(df1[x + j][10])
                j_policy = int(df1[x + j][12])
                if (j == 0):
                    max_policy = j_policy
                if (j_util > last_util):
                    max_idx = j - 1
                    max_policy = j_policy
                last_util = j_util
                cpu_util_list.append(j_cpu_util)
            # 计算基于原始util的频率
            max_j_util = max(cpu_util_list)
            max_freq = int(df1[x + 1][16])
            raw_freq = int(max_j_util * 1.25 * max_freq / 1024)
            util_to_freq = s_big_freq[np.searchsorted(s_big_freq, raw_freq)] if s_big_freq[0] <= raw_freq <= s_big_freq[
                -1] \
                else (s_big_freq[0] if raw_freq < s_big_freq[0] else s_big_freq[-1])
            super_j_util_freq_list[s_big_freq.index(util_to_freq)] = super_j_util_freq_list[
                                                                         s_big_freq.index(util_to_freq)] + 1

            # 计算基于最终经过boost的util的频率
            raw_freq = int(last_util * 1.25 * max_freq / 1024)
            freq = s_big_freq[np.searchsorted(s_big_freq, raw_freq)] if s_big_freq[0] <= raw_freq <= s_big_freq[-1] \
                else (s_big_freq[0] if raw_freq < s_big_freq[0] else s_big_freq[-1])
            # super_freq_policy_list[s_big_freq.index(freq)].append(max_policy)
            # 在sugov统计的频率记为req_freq
            normal_freq = int(df1[x + 1][12])
            req_freq = int(df1[x + 1][14])
            # print(req_freq, freq)
            if normal_freq == req_freq:
                if (normal_freq == freq):
                    if (util_to_freq < freq):
                        super_count = super_count + 1
                        super_freq_policy_list[s_big_freq.index(req_freq)].append(max_policy)
                    else:
                        # print(1)
                        super_count = super_count + 1
                        super_freq_policy_list[s_big_freq.index(req_freq)].append(0)

                else:
                    super_freq_policy_list[s_big_freq.index(req_freq)].append(-1)
            else:
                super_freq_policy_list[s_big_freq.index(req_freq)].append(-2)
            super_freq_list[s_big_freq.index(req_freq)] = super_freq_list[s_big_freq.index(req_freq)] + 1

    sum_freq_count = np.sum(small_freq_list) + np.sum(big_freq_list) + np.sum(super_freq_list)
    for i in range(16):
        small_freq_list[i] = small_freq_list[i] / sum_freq_count
        big_freq_list[i] = big_freq_list[i] / sum_freq_count
        small_j_util_freq_list[i] = small_j_util_freq_list[i] / sum_freq_count
        big_j_util_freq_list[i] = big_j_util_freq_list[i] / sum_freq_count
    for i in range(19):
        super_freq_list[i] = super_freq_list[i] / sum_freq_count
        super_j_util_freq_list[i] = super_j_util_freq_list[i] / sum_freq_count

    print(small_count, small_total_count, big_count, big_total_count)
    for i in range(16):
        super_policy_result = count_and_sort(super_freq_policy_list[i])
        print(super_policy_result)

    small_policy_counter = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(16)]
    big_policy_counter = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(16)]
    super_policy_counter = [[0, 0, 0, 0, 0, 0, 0, 0] for _ in range(19)]

    small_cpu_counter = [[0, 0, 0, 0] for _ in range(16)]
    big_cpu_counter = [[0, 0, 0] for _ in range(16)]

    for i in range(16):
        for j in range(len(small_freq_policy_list[i])):
            # 记录这一次选频最终决定频率的cpu
            cpu = small_freq_cpu_list[i][j]
            small_cpu_counter[i][cpu] = small_cpu_counter[i][cpu] + 1

            policy = small_freq_policy_list[i][j]
            thousand, hundred, ten, unit = check_digits_for_1(policy)
            if thousand:
                small_policy_counter[i][0] = small_policy_counter[i][0] + 1
            if hundred:
                small_policy_counter[i][1] = small_policy_counter[i][1] + 1
            if ten:
                small_policy_counter[i][2] = small_policy_counter[i][2] + 1
            if unit:
                small_policy_counter[i][3] = small_policy_counter[i][3] + 1
            if policy == 0:
                small_policy_counter[i][4] = small_policy_counter[i][4] + 1
            if policy == -1:
                small_policy_counter[i][5] = small_policy_counter[i][5] + 1
            if policy == -2:
                small_policy_counter[i][6] = small_policy_counter[i][6] + 1
        for j in range(len(big_freq_policy_list[i])):
            # 记录这一次选频最终决定频率的cpu
            cpu = big_freq_cpu_list[i][j]
            big_cpu_counter[i][cpu] = big_cpu_counter[i][cpu] + 1

            policy = big_freq_policy_list[i][j]
            thousand, hundred, ten, unit = check_digits_for_1(policy)
            if thousand:
                big_policy_counter[i][0] = big_policy_counter[i][0] + 1
            if hundred:
                big_policy_counter[i][1] = big_policy_counter[i][1] + 1
            if ten:
                big_policy_counter[i][2] = big_policy_counter[i][2] + 1
            if unit:
                big_policy_counter[i][3] = big_policy_counter[i][3] + 1
            if policy == 0:
                big_policy_counter[i][4] = big_policy_counter[i][4] + 1
            if policy == -1:
                big_policy_counter[i][5] = big_policy_counter[i][5] + 1
            if policy == -2:
                big_policy_counter[i][6] = big_policy_counter[i][6] + 1
    for i in range(19):
        for j in range(len(super_freq_policy_list[i])):
            policy = super_freq_policy_list[i][j]
            thousand, hundred, ten, unit = check_digits_for_1(policy)
            if thousand:
                super_policy_counter[i][0] = super_policy_counter[i][0] + 1
            if hundred:
                super_policy_counter[i][1] = super_policy_counter[i][1] + 1
            if ten:
                super_policy_counter[i][2] = super_policy_counter[i][2] + 1
            if unit:
                super_policy_counter[i][3] = super_policy_counter[i][3] + 1
            if policy == 0:
                super_policy_counter[i][4] = super_policy_counter[i][4] + 1
            if policy == -1:
                super_policy_counter[i][5] = super_policy_counter[i][5] + 1
            if policy == -2:
                super_policy_counter[i][6] = super_policy_counter[i][6] + 1
    for x in range(16):
        if np.sum(small_policy_counter[x]) == 0:
            # print(x)
            small_policy_counter[x][7] = 1
        if np.sum(big_policy_counter[x]) == 0:
            # print(x)
            big_policy_counter[x][7] = 1
    for x in range(19):
        if np.sum(super_policy_counter[x]) == 0:
            # print(x)
            super_policy_counter[x][7] = 1
    print(super_policy_counter)
    print(super_total_count, super_count)
    plt_boost_policy(small_policy_counter, 'small')
    plt_boost_policy(big_policy_counter, 'big')
    plt_boost_policy(super_policy_counter, 'super')

    plt_boost_policy(small_cpu_counter, 'cpu_small')
    plt_boost_policy(big_cpu_counter, 'cpu_big')
    return [small_j_util_freq_list, small_freq_list], [big_j_util_freq_list, big_freq_list], [super_j_util_freq_list,
                                                                                              super_freq_list]


if __name__ == '__main__':
    #  analysis_freq_trace('./trace.txt')
    # analysis_freq_contribute(r'./freq_details.csv', r'./freq_contribute.csv')
    plot_freq_contribute(r'./freq_contribute.csv')
