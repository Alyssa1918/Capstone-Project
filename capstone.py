import numpy as np
import pandas as pd
import scipy.optimize as opt

def load_data(location):
    df = pd.read_excel(rf"{location}", header=None)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    return df.to_numpy()

def section_df(df):
    sectioned_df = []
    for i in range(3,20):
        temp_df = []
        for j in range(len(df)):
            temp_df.append(df[j][0:i])
        sectioned_df.append(temp_df)
    return sectioned_df

def get_ts_df(all_t1_indices, all_t2_indices, all_t3_indices, df):
    t1_df = []
    t2_df = []
    t3_df = []
    count = 0
    for i in range(len(all_t1_indices)):
        t1_df.append(df[count][:all_t1_indices[i]])
        t2_df.append(df[count][:all_t2_indices[i]])
        t3_df.append(df[count][:all_t3_indices[i]])
        count += 1
        t1_df.append(df[count][:all_t1_indices[i]])
        t2_df.append(df[count][:all_t2_indices[i]])
        t3_df.append(df[count][:all_t3_indices[i]])
        count += 1

    return (t1_df, t2_df, t3_df)

def get_y(df):
    y = []
    count = 0
    for row in df:
        count += 1
        if count % 2 == 1:
            continue
        temp = row[~np.isnan(row)]
        y.append(np.log(temp))
    return y
        
def get_phi(df):
    phi1 = []
    phi2 = []
    phi3 = []
    count = 0
    for row in df:
        count += 1
        if count % 2 == 0:
            continue
        else:
            temp = row[~np.isnan(row)]
            phi1.append(np.ones(temp.shape))
            phi2.append(np.log(temp))
            phi3.append(temp)
    return phi1, phi2, phi3

def get_times(df):
    times = []
    count = 0
    for row in df:
        count += 1
        if count % 2 == 0:
            continue
        temp = row[~np.isnan(row)]
        times.append(temp)
    return times

def get_concs(df):
    concs = []
    count = 0
    for row in df:
        count += 1
        if count % 2 == 1:
            continue
        temp = row[~np.isnan(row)]
        concs.append(temp)
    return concs

def get_ts(all_concs):
    # find t1
    all_t1_indices = []
    for concs in all_concs:
        t1_index = np.argmax(concs)
        all_t1_indices.append(t1_index)

    all_t1_concs = []
    for i, concs in enumerate(all_concs):
        all_t1_concs.append(concs[all_t1_indices[i]])

    # find t2 and t3
    potential_t2_concs = .5 * np.array(all_t1_concs)
    potential_t3_concs = .25 * np.array(all_t1_concs)

    all_t2_indices = []
    all_t3_indices = []
    for i, convs in enumerate(all_concs):
        all_t2_indices.append(np.argmax(convs[all_t1_indices[i] + 1:] <= potential_t2_concs[i]) + all_t1_indices[i] + 1)
        all_t3_indices.append(np.argmax(convs[all_t1_indices[i] + 1:] <= potential_t3_concs[i]) + all_t1_indices[i] + 1)

    return all_t1_indices, all_t2_indices, all_t3_indices

def calc_min(y, phi1, phi2, phi3):
    all_u = []
    all_z = []
    all_delta = []
    for i,_ in enumerate(y):
        c = np.concatenate((np.zeros([3,1]), np.ones([len(phi1[i]), 1])), axis=0)
        A1 = np.concatenate((phi1[i].reshape(-1,1), phi2[i].reshape(-1,1), phi3[i].reshape(-1,1), -np.eye(len(phi1[i]))), axis=1)
        A2 = np.concatenate((-phi1[i].reshape(-1,1), -phi2[i].reshape(-1,1), -phi3[i].reshape(-1,1), -np.eye(len(phi1[i]))), axis=1)
        A = np.concatenate((A1, A2), axis=0)
        b = np.concatenate((y[i].reshape(-1,1),-y[i].reshape(-1,1)), axis=0)
        bds = (((-np.inf, np.inf), )*(len(phi1[i])+3))
    
        res = opt.linprog(c, A, b, bounds=bds)

        all_u.append(res['x'][:3])
        all_z.append(res['fun'])
        all_delta.append(res['x'][3:])
    return all_u, all_delta, all_z

def get_all_abc(all_u):
    all_abc = []
    for row in all_u:
        a = np.e**row[0]
        b = row[1]
        c = -1/row[2]
        all_abc.append(np.array([a,b,c]))
    return all_abc

def calc_concs_diff_avg(all_times, all_abc, all_concs):
    pred_concs = []
    for i,times in enumerate(all_times):
        conc = all_abc[i][0] * times**all_abc[i][1] * np.e**(-times/all_abc[i][2])
        pred_concs.append(conc)
    
    diffs = []
    for i in range(len(all_concs)):
        diffs.extend(pred_concs[i] - all_concs[i])
    diffs = np.abs(diffs)
    avg = np.average(diffs)

    return avg


def calc_ts_diff_avg(all_t1_indices, all_t2_indices, all_t3_indices, all_times, all_abc, all_concs):
    pred_concs1 = []
    pred_concs2 = []
    pred_concs3 = []
    for i,times in enumerate(all_times):
        conc = all_abc[i][0] * times**all_abc[i][1] * np.e**(-times/all_abc[i][2])
        conc1 = conc[all_t1_indices[i]]
        conc2 = conc[all_t2_indices[i]]
        conc3 = conc[all_t3_indices[i]]
        pred_concs1.append(conc1)
        pred_concs2.append(conc2)
        pred_concs3.append(conc3)

    concs1 = []
    concs2 = []
    concs3 = []
    for i,times in enumerate(all_times):
        concs1.append(all_concs[i][all_t1_indices[i]])
        concs2.append(all_concs[i][all_t2_indices[i]])
        concs3.append(all_concs[i][all_t3_indices[i]])

    diff1 = []
    diff2 = []
    diff3 = []
    diff1.extend(np.array(concs1) - np.array(pred_concs1))
    diff2.extend(np.array(concs2) - np.array(pred_concs2))
    diff3.extend(np.array(concs3) - np.array(pred_concs3))
    avg1 = np.average(np.abs(np.array(diff1)))
    avg2 = np.average(np.abs(np.array(diff2)))
    avg3 = np.average(np.abs(np.array(diff3)))

    return avg1, avg2, avg3
    

def get_avg_t2_time(all_t2_indices, all_times):
    all_t2_times = []
    for i,times in enumerate(all_times):
        all_t2_times.append(times[all_t2_indices[i]])
    return np.average(np.array(all_t2_times))


def print_avgs_and_ts_avgs(avgs, ts_avg):
    print("Section Number | Avg Diff | Avg Diff at t1 | Avg Diff at t2 | Avg Diff at t3")
    for i in range(len(avgs)):
        print(f"Section {i + 1} | {avgs[i]:.5f} | {ts_avg[i][0]:.5f} | {ts_avg[i][1]:.5f} | {ts_avg[i][2]:.5f}")


def print_ts_avgs_and_ts_ts_avgs(ts_avgs, ts_ts_avg):
    print("ts Number | Avg Diff | Avg Diff at t1 | Avg Diff at t2 | Avg Diff at t3")
    for i in range(len(ts_avgs)):
        print(f"t{i + 1} | {ts_avgs[i]:.5f} | {ts_ts_avg[i][0]:.5f} | {ts_ts_avg[i][1]:.5f} | {ts_ts_avg[i][2]:.5f}")


def print_ts(all_t1_indices, all_t2_indices, all_t3_indices):
    print("Patient Number | t1 Index | t2 Index | t3 Index")
    for i in range(len(all_t1_indices)):
        print(f"{i + 1} | {all_t1_indices[i]} | {all_t2_indices[i]} | {all_t3_indices[i]} ")


if __name__ == "__main__":
    df = load_data(r"C:\Users\alyla\OneDrive\Documents\Math 464\SlipStreamData.xlsx")
    sectioned_df = section_df(df)
    all_times = get_times(df)
    all_concs = get_concs(df)
    all_t1_indices, all_t2_indices, all_t3_indices = get_ts(all_concs)
    avgs = []
    ts_avg = []
    for section in sectioned_df:
        y = get_y(section)
        phi1, phi2, phi3 = get_phi(section)
        all_u, all_delta, all_z = calc_min(y, phi1, phi2, phi3)
        all_abc = get_all_abc(all_u)
        avg = calc_concs_diff_avg(all_times, all_abc, all_concs)
        t1_avg, t2_avg, t3_avg = calc_ts_diff_avg(all_t1_indices, all_t2_indices, all_t3_indices, all_times, all_abc, all_concs)
        avgs.append(avg)
        ts_avg.append((t1_avg, t2_avg, t3_avg))
    print_avgs_and_ts_avgs(avgs, ts_avg)
    print()
    print_ts(all_t1_indices, all_t2_indices, all_t3_indices)

    ts_df = get_ts_df(all_t1_indices, all_t2_indices, all_t3_indices, df)
    ts_avgs = []
    ts_ts_avg = []
    for t_df in ts_df:
        y = get_y(t_df)
        phi1, phi2, phi3 = get_phi(t_df)
        all_u, all_delta, all_z = calc_min(y, phi1, phi2, phi3)
        all_abc = get_all_abc(all_u)
        avg = calc_concs_diff_avg(all_times, all_abc, all_concs)
        t1_avg, t2_avg, t3_avg = calc_ts_diff_avg(all_t1_indices, all_t2_indices, all_t3_indices, all_times, all_abc, all_concs)
        ts_avgs.append(avg)
        ts_ts_avg.append((t1_avg, t2_avg, t3_avg))
    print()
    print_ts_avgs_and_ts_ts_avgs(ts_avgs, ts_ts_avg)

    avg_t2_time = get_avg_t2_time(all_t2_indices, all_times)
    print(f"\nAverage t2 Time: {avg_t2_time}")
    


