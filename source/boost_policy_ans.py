import warnings
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


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


def read_boost_next_freq(file_name, app, ii, itemp):
    global freq
    import codecs

    with codecs.open(r'data_trace/{}/{}_{}_{}/trace.txt'.format(file_name, app, ii, itemp), 'rb') as file:
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
