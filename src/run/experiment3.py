import utils.utils
from config import configuration
from experiments import experiment3_gpt2, experiment3_ngrams
import statistics
import os

from utils.utils import print_title


def main():
    utils.utils.print_title("Experiment 3")

    # Run experiment 3 for GPT2 models
    for epochs in[0,5,30,20,30]:
        experiment3_gpt2.run_experiment(epochs)

    # Run experiment 3 for N-gram models
    for order in [2,3,4,5]:
        experiment3_ngrams.run_experiment(order)

    for fn in os.listdir(configuration.experiment3_output_base_path):
        if fn != "." and fn != "..":
            eval_perplexity_scores(configuration.experiment3_output_base_path + fn)

def eval_perplexity_scores(file_path):
    control_dict = {}
    dementia_dict = {}

    # ---------------------------------------------------
    # --- 			Fill patients dictionary		  ---
    # ---------------------------------------------------
    if "=" in file_path:
        epochs = file_path.split("=")[-1].split(".")[0]
    else:
        epochs = file_path.split("-")[0]
    with open(file_path, 'r') as f:
        file_content = f.read().split("\n")

    for line in file_content:
        line = line.split(",")
        if line[0] != "Group Id" and line[0] != "":
            group_id = line[0]
            subject_id = line[1]
            interview_id = line[2]
            p_control = float(line[3])
            p_dementia = float(line[4])

            # 4{
            #   'p_control': [...],
            #   'p_control': [...],
            #   'devstd_control': [...],
            #   'devstd_dementia': [...]
            # }
            if group_id == 'dementia':
                if subject_id not in dementia_dict:
                    dementia_dict[subject_id] = {}
                if 'p_control' not in dementia_dict[subject_id]:
                    dementia_dict[subject_id]['p_control'] = []
                if 'p_dementia' not in dementia_dict[subject_id]:
                    dementia_dict[subject_id]['p_dementia'] = []
                if 'c-d' not in dementia_dict[subject_id]:
                    dementia_dict[subject_id]['c-d'] = []
                if 'd-c' not in dementia_dict[subject_id]:
                    dementia_dict[subject_id]['d-c'] = []

                dementia_dict[subject_id]['p_control'].append(p_control)
                dementia_dict[subject_id]['p_dementia'].append(p_dementia)
                dementia_dict[subject_id]['c-d'].append(p_control - p_dementia)
                dementia_dict[subject_id]['d-c'].append(p_dementia - p_control)

            if group_id == 'control':
                if subject_id not in control_dict:
                    control_dict[subject_id] = {}
                if 'p_control' not in control_dict[subject_id]:
                    control_dict[subject_id]['p_control'] = []
                if 'p_dementia' not in control_dict[subject_id]:
                    control_dict[subject_id]['p_dementia'] = []
                if 'c-d' not in control_dict[subject_id]:
                    control_dict[subject_id]['c-d'] = []
                if 'd-c' not in control_dict[subject_id]:
                    control_dict[subject_id]['d-c'] = []

                control_dict[subject_id]['p_control'].append(p_control)
                control_dict[subject_id]['p_dementia'].append(p_dementia)
                control_dict[subject_id]['c-d'].append(p_control - p_dementia)
                control_dict[subject_id]['d-c'].append(p_dementia - p_control)

    # Estimating thresholds with leave one out
    for s in control_dict:
        p_control_avg = []
        p_dementia_avg = []
        dc_avg = []
        cd_avg = []

        for s1 in control_dict:
            if s != s1:
                p_control_avg.append(statistics.mean(control_dict[s1]['p_control']))
                p_dementia_avg.append(statistics.mean(control_dict[s1]['p_dementia']))
                cd_avg.append(statistics.mean(control_dict[s1]['c-d']))
                dc_avg.append(statistics.mean(control_dict[s1]['d-c']))

        control_dict[s]['p_control_avg'] = statistics.mean(p_control_avg)
        control_dict[s]['p_control_std'] = statistics.stdev(p_control_avg)

        control_dict[s]['p_dementia_avg'] = statistics.mean(p_dementia_avg)
        control_dict[s]['p_dementia_std'] = statistics.stdev(p_dementia_avg)

        control_dict[s]['d-c_avg'] = statistics.mean(dc_avg)
        control_dict[s]['d-c_std'] = statistics.stdev(dc_avg)

        control_dict[s]['c-d_avg'] = statistics.mean(cd_avg)
        control_dict[s]['c-d_std'] = statistics.stdev(cd_avg)

    for s in dementia_dict:
        p_control_avg = []
        p_dementia_avg = []
        dc_avg = []
        cd_avg = []

        for s1 in dementia_dict:
            if s != s1:
                p_control_avg.append(statistics.mean(dementia_dict[s1]['p_control']))
                p_dementia_avg.append(statistics.mean(dementia_dict[s1]['p_dementia']))
                cd_avg.append(statistics.mean(dementia_dict[s1]['c-d']))
                dc_avg.append(statistics.mean(dementia_dict[s1]['d-c']))

        dementia_dict[s]['p_control_avg'] = statistics.mean(p_control_avg)
        dementia_dict[s]['p_control_std'] = statistics.stdev(p_control_avg)

        dementia_dict[s]['p_dementia_avg'] = statistics.mean(p_dementia_avg)
        dementia_dict[s]['p_dementia_std'] = statistics.stdev(p_dementia_avg)

        dementia_dict[s]['d-c_avg'] = statistics.mean(dc_avg)
        dementia_dict[s]['d-c_std'] = statistics.stdev(dc_avg)

        dementia_dict[s]['c-d_avg'] = statistics.mean(cd_avg)
        dementia_dict[s]['c-d_std'] = statistics.stdev(cd_avg)

    # ===================================================

    print_title("Results for " + file_path.split("/")[-1])
    print("\n", "=" * 80)
    count_control = 0
    count_dementia = 0
    for l in control_dict.values():
        count_control += + len(l['p_control'])
    for l in dementia_dict.values():
        count_dementia += + len(l['p_control'])
    print("# subjects for control: ", len(control_dict))
    print("# subjects for dementia: ", len(dementia_dict))
    print("# interviews for control: ", count_control)
    print("# interviews for dementia: ", count_dementia)
    print()
    # ===================================================

    # ---------------------------------------------------
    # ---  			 Compute control avg   		      ---
    # ---------------------------------------------------
    control_control_ppl_list = []
    control_dementia_ppl_list = []
    control_diff_cd_ppl_list = []
    control_diff_dc_ppl_list = []
    for subject_id in control_dict:
        control_control_ppl_list.extend(control_dict[subject_id]['p_control'])
        control_dementia_ppl_list.extend(control_dict[subject_id]['p_dementia'])
        control_diff_cd_ppl_list.extend(control_dict[subject_id]['c-d'])
        control_diff_dc_ppl_list.extend(control_dict[subject_id]['d-c'])

    # PPL media del gruppo controllo assegnata dal gruppo controllo
    control_control_avg = statistics.mean(control_control_ppl_list)
    control_control_std = statistics.stdev(control_control_ppl_list)

    # PPL media del gruppo controllo assegnata dal gruppo demenza
    control_dementia_avg = statistics.mean(control_dementia_ppl_list)
    control_dementia_std = statistics.stdev(control_dementia_ppl_list)

    # Differenza media c-d e d-c per gruppo controllo
    control_diff_cd_avg = statistics.mean(control_diff_cd_ppl_list)
    control_diff_cd_std = statistics.stdev(control_diff_cd_ppl_list)
    #
    control_diff_dc_avg = statistics.mean(control_diff_dc_ppl_list)
    control_diff_dc_std = statistics.stdev(control_diff_dc_ppl_list)

    # ---------------------------------------------------
    # ---            Compute dementia avg             ---
    # ---------------------------------------------------
    dementia_control_ppl_list = []
    dementia_dementia_ppl_list = []
    dementia_diff_cd_ppl_list = []
    dementia_diff_dc_ppl_list = []
    for subject_id in dementia_dict:
        dementia_control_ppl_list.extend(dementia_dict[subject_id]['p_control'])
        dementia_dementia_ppl_list.extend(dementia_dict[subject_id]['p_dementia'])
        dementia_diff_cd_ppl_list.extend(dementia_dict[subject_id]['c-d'])
        dementia_diff_dc_ppl_list.extend(dementia_dict[subject_id]['d-c'])

    # PPL media del gruppo demenza assegnata dal gruppo controllo
    dementia_control_avg = statistics.mean(dementia_control_ppl_list)
    dementia_control_std = statistics.stdev(dementia_control_ppl_list)

    # PPL media del gruppo demenza assegnata dal gruppo demenza
    dementia_dementia_avg = statistics.mean(dementia_dementia_ppl_list)
    dementia_dementia_std = statistics.stdev(dementia_dementia_ppl_list)

    # Differenza media c-d e d-c per gruppo demenza
    dementia_diff_cd_avg = statistics.mean(dementia_diff_cd_ppl_list)
    dementia_diff_cd_std = statistics.stdev(dementia_diff_cd_ppl_list)
    #
    dementia_diff_dc_avg = statistics.mean(dementia_diff_dc_ppl_list)
    dementia_diff_dc_std = statistics.stdev(dementia_diff_dc_ppl_list)

    # ---------------------------------------------------
    print("Control group AVG Ppl from Control:", control_control_avg)
    print("Control group AVG Ppl from Dementia:", control_dementia_avg)
    print("Control group AVG Ppl difference [Control-Dementia]:", control_diff_cd_avg)
    print("Control group AVG Ppl difference [Dementia-Control]:", control_diff_dc_avg)
    print("Control group STD.DEV Ppl difference [Dementia-Control]:", control_diff_dc_std)

    print()

    print("Dementia group AVG Ppl from Control:", dementia_control_avg)
    print("Dementia group AVG Ppl from Dementia:", dementia_dementia_avg)
    print("Dementia group AVG Ppl difference [Control-Dementia]:", dementia_diff_cd_avg)
    print("Dementia group AVG Ppl difference [Dementia-Control]:", dementia_diff_dc_avg)

    print("-" * 50, "\n")

    # --- Control
    classify_patients_with_threshold(control_dict, dementia_dict, control_control_avg, 'p_control', '<', '>')

    # --- Dementia
    classify_patients_with_threshold(control_dict, dementia_dict, dementia_dementia_avg, 'p_dementia', '>', '<')

    # --- D-C and (D-C)^* ---
    classify_patients_with_threshold_dev(control_dict, dementia_dict, control_diff_dc_avg, control_diff_dc_std, dementia_diff_dc_avg, dementia_diff_dc_std, 'd-c', 0)
    classify_patients_with_threshold_dev(control_dict, dementia_dict, control_diff_dc_avg, control_diff_dc_std, dementia_diff_dc_avg, dementia_diff_dc_std, 'd-c', 2)

def classify_patients_with_threshold(control_dict, dementia_dict, threshold, criterion, control_rule, dementia_rule):

    # ---------------------------------------------------
    # ---       Count dementia recognized patients    ---
    # ---------------------------------------------------

    tpd = 0.0
    fpd = 0.0
    tnd = 0.0
    fnd = 0.0

    tpc = 0.0
    fpc = 0.0
    tnc = 0.0
    fnc = 0.0

    subjects = {}
    for s in dementia_dict:
        subjects['d_' + s] = dementia_dict[s]
    for s in control_dict:
        subjects['c_' + s] = control_dict[s]

    for s in subjects:
        subject_avg_ppl = statistics.mean(subjects[s][criterion])
        group = s.split("_")[0]

        if group == 'd':
            if subject_avg_ppl > threshold:
                if dementia_rule == '>':
                    tpd += 1.0
                    tnc += 1.0
                else:
                    fnd += 1.0
                    fpc += 1.0
            else:
                if dementia_rule == '>':
                    fnd += 1.0
                    fpc += 1.0
                else:
                    tpd += 1.0
                    tnc += 1.0
        else:
            if subject_avg_ppl > threshold:
                if control_rule == '>':
                    tpc += 1.0
                    tnd += 1.0
                else:
                    fnc += 1.0
                    fpd += 1.0
            else:
                if control_rule == '>':
                    fnc += 1.0
                    fpd += 1.0
                else:
                    tpc += 1.0
                    tnd += 1.0

    # ===================================================

    print("#" * 52)
    print("#" * 20 + " Statistics " + "#" * 20)
    print("#" * 52)
    print("Threshold:", threshold)
    print("Criterion: ", criterion)
    print("Control rule:", control_rule)
    print("Dementia rule:", dementia_rule)
    print()
    print("Detecting Dementia")
    print("TP:", tpd)
    print("FP:", fpd)
    print("FN:", fnd)
    print("TN:", tnd)
    precision = tpd / (tpd + fpd)
    recall = tpd / (tpd + fnd)
    print("P:", precision)
    print("R:", recall)
    print("F1:", 2 * (precision * recall) / (precision + recall))
    print()
    print("Accuracy:", (tpd + tnd) / (tpd + tnd + fpd + fnd))
    print()
    print("Detecting Control")
    print("TP:", tpc)
    print("FP:", fpc)
    print("FN:", fnc)
    print("TN:", tnc)
    precision = 0 if tpc + fpc == 0 else tpc / (tpc + fpc)
    recall = 0 if tpc + fnc == 0 else tpc / (tpc + fnc)
    print("P:", precision)
    print("R:", recall)
    print("F1:", 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall))
    print()

def classify_patients_with_threshold_dev(control_dict, dementia_dict, tc, devc, td, devd, criterion, m):

    # ---------------------------------------------------
    # ---       Count dementia recognized patients    ---
    # ---------------------------------------------------

    tpd = 0.0
    fpd = 0.0
    tnd = 0.0
    fnd = 0.0

    tpc = 0.0
    fpc = 0.0
    tnc = 0.0
    fnc = 0.0

    subjects = {}
    for s in dementia_dict:
        subjects['d_' + s] = dementia_dict[s]
    for s in control_dict:
        subjects['c_' + s] = control_dict[s]

    for s in subjects:
        ttd = td
        ddevd = devd
        ttc = tc
        ddevc = devc

        subject_avg_ppl = statistics.mean(subjects[s][criterion])
        group = s.split("_")[0]
        subject = s.split("_")[1]
        # \text{if } PPL(i) > \left(\text{avg(D-C)}_{C} - 2\cdot\text{stdev}_{C} \right)\text{ then } C\\
        # \text{if } PPL(i) < \left(\text{avg(D-C)}_{D} + 2\cdot\text{stdev}_{D} \right)\text{ then } D

        # print("-"*30)
        # print(ttc,ddevc,ttd,ddevd)
        if group == 'd':
            ttd = dementia_dict[subject][criterion + '_avg']
            ddevd = dementia_dict[subject][criterion + '_std']
        else:
            ttc = control_dict[subject][criterion + '_avg']
            ddevc = control_dict[subject][criterion + '_std']
        # print(ttc,ddevc,ttd,ddevd)

        if subject_avg_ppl < ttd + m * ddevd:
            if group == 'd':
                tpd += 1.0
            else:
                fpd += 1.0
        else:
            if group == 'd':
                fnd += 1.0
            else:
                tnd += 1.0

        if subject_avg_ppl > ttc - m * ddevc:
            if group == 'c':
                tpc += 1.0
            else:
                fpc += 1.0
        else:
            if group == 'c':
                fnc += 1.0
            else:
                tnc += 1.0

    # ===================================================

    print("#" * 52)
    print("#" * 20 + " Statistics " + "#" * 20)
    print("#" * 52)
    print("Tc:", tc)
    print("Devc: ", devc)
    print("Td:", td)
    print("Devd: ", devd)
    print("Multiplier: ", m)
    print()
    print("Detecting Dementia")
    print("TP:", tpd)
    print("FP:", fpd)
    print("FN:", fnd)
    print("TN:", tnd)
    precision = tpd / (tpd + fpd)
    recall = tpd / (tpd + fnd)
    print("P:", precision)
    print("R:", recall)
    print("F1:", 2 * (precision * recall) / (precision + recall))
    print()
    print("Accuracy:", (tpd + tnd) / (tpd + tnd + fpd + fnd))
    print()
    print("Detecting Control")
    print("TP:", tpc)
    print("FP:", fpc)
    print("FN:", fnc)
    print("TN:", tnc)
    precision = 0 if tpc + fpc == 0 else tpc / (tpc + fpc)
    recall = 0 if tpc + fnc == 0 else tpc / (tpc + fnc)
    print("P:", precision)
    print("R:", recall)
    print("F1:", 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall))
    print()


if __name__ == '__main__':
    main()