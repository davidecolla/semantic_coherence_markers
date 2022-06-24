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

    # ../resources/data/experiment3/output/[2-grams.csv,3-grams.csv, ...]
    out_csv = "Model, Rule, Accuracy, Pd, Rd, F1d, Pc, Rc, F1c, HM,Dc>Dad,D^*_c>D^*_ad,Dc,DEVc,Dad,DEVad\n"
    for fn in os.listdir(configuration.experiment3_output_base_path):
        if fn != "." and fn != ".." and "results" not in fn:
            results = eval_perplexity_scores(configuration.experiment3_output_base_path + fn)
            out_csv += results + "\n"
    open(configuration.experiment3_output_base_path + "results.csv", "w").write(out_csv)

def eval_perplexity_scores(file_path):
    control_dict = {}
    dementia_dict = {}
    results = ""

    # ---------------------------------------------------
    # --- 			Fill patients dictionary		  ---
    # ---------------------------------------------------
    if "=" in file_path:
        epochs = file_path.split("=")[-1].split(".")[0]
    else:
        epochs = file_path.split("-")[0]

    # Read file
    file_content = open(file_path, 'r').read().split("\n")

    for line in file_content:
        line = line.split(",")
        if line[0] != "Group Id" and line[0] != "":
            # Load data from file
            group_id = line[0]
            subject_id = line[1]
            interview_id = line[2]
            p_control = float(line[3])
            p_dementia = float(line[4])

            # Final dictionaries, one for each group, will be in the following form:
            # 4{
            #   'p_control': [...],
            #   'p_control': [...],
            #   'c-d': [...],
            #   'd-c': [...],
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

                # Load PPL scores together with differences for the AD group
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

                # Load PPL scores together with differences for the Control group
                control_dict[subject_id]['p_control'].append(p_control)
                control_dict[subject_id]['p_dementia'].append(p_dementia)
                control_dict[subject_id]['c-d'].append(p_control - p_dementia)
                control_dict[subject_id]['d-c'].append(p_dementia - p_control)

    # --------------------------------- #
    #      Estimating thresholds        #
    # --------------------------------- #
    #
    # For each subject belonging to the Control group we estimate the thresholds in a leave one out setting.
    # More precisely, we compute average en standard deviations of all the scores for patients in the Control group
    # except for s.
    # The subject s will be the held-out subject.
    for s in control_dict:
        p_control_avg = []
        p_dementia_avg = []
        dc_avg = []
        cd_avg = []

        # For each subject s1 in the Control group, if s1 != s, then add her/his scores to our lists.
        for s1 in control_dict:
            if s != s1:
                p_control_avg.append(statistics.mean(control_dict[s1]['p_control']))
                p_dementia_avg.append(statistics.mean(control_dict[s1]['p_dementia']))
                cd_avg.append(statistics.mean(control_dict[s1]['c-d']))
                dc_avg.append(statistics.mean(control_dict[s1]['d-c']))

        assert len(dc_avg) == len(control_dict)-1 and len(cd_avg) == len(control_dict)-1
        # Compute average and st.dev from the lists
        control_dict[s]['p_control_avg'] = statistics.mean(p_control_avg)
        control_dict[s]['p_control_std'] = statistics.stdev(p_control_avg)

        control_dict[s]['p_dementia_avg'] = statistics.mean(p_dementia_avg)
        control_dict[s]['p_dementia_std'] = statistics.stdev(p_dementia_avg)

        control_dict[s]['d-c_avg'] = statistics.mean(dc_avg)
        control_dict[s]['d-c_std'] = statistics.stdev(dc_avg)

        control_dict[s]['c-d_avg'] = statistics.mean(cd_avg)
        control_dict[s]['c-d_std'] = statistics.stdev(cd_avg)

    # For each subject belonging to the AD group we estimate the thresholds in a leave one out setting.
    # More precisely, we compute average en standard deviations of all the scores for patients in the AD group
    # except for s.
    # The subject s will be the held-out subject.
    for s in dementia_dict:
        p_control_avg = []
        p_dementia_avg = []
        dc_avg = []
        cd_avg = []

        # For each subject s1 in the AD group, if s1 != s, then add her/his scores to our lists.
        for s1 in dementia_dict:
            if s != s1:
                p_control_avg.append(statistics.mean(dementia_dict[s1]['p_control']))
                p_dementia_avg.append(statistics.mean(dementia_dict[s1]['p_dementia']))
                cd_avg.append(statistics.mean(dementia_dict[s1]['c-d']))
                dc_avg.append(statistics.mean(dementia_dict[s1]['d-c']))

        assert len(dc_avg) == len(dementia_dict) - 1 and len(cd_avg) == len(dementia_dict) - 1
        # Compute average and st.dev from the lists
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
    # We compute the global thresholds by including scores from all the patients
    # belonging to the Control group.
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
    # We compute the global thresholds by including scores from all the patients
    # belonging to the AD group.
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
    # return

    print()

    print("Dementia group AVG Ppl from Control:", dementia_control_avg)
    print("Dementia group AVG Ppl from Dementia:", dementia_dementia_avg)
    print("Dementia group AVG Ppl difference [Control-Dementia]:", dementia_diff_cd_avg)
    print("Dementia group AVG Ppl difference [Dementia-Control]:", dementia_diff_dc_avg)

    print()

    print("### Global thresholds ###")
    print("Control group Diff. AVG:", control_diff_dc_avg)
    print("Control group Diff. Dev.STD:", control_diff_dc_std)
    print("Dementia group Diff. AVG:", dementia_diff_dc_avg)
    print("Dementia group Diff. Dev.STD:", dementia_diff_dc_std)

    print("-" * 50, "\n")

    # --- Control ---
    acc,pd,rd,fd,pc,rc,fc = classify_patients_with_threshold(control_dict, dementia_dict, control_control_avg, 'p_control', '<', '>')
    hm = statistics.harmonic_mean([acc,fd,fc])
    results += file_path.split("/")[-1] + ",Pc," + str(acc) + "," + str(pd) + "," + str(rd) + "," + str(fd) + "," + str(pc) + "," + str(rc) + "," + str(fc) + ","  + str(hm) + "\n"
    # --- Dementia ---
    acc,pd,rd,fd,pc,rc,fc = classify_patients_with_threshold(control_dict, dementia_dict, dementia_dementia_avg, 'p_dementia', '>', '<')
    hm = statistics.harmonic_mean([acc,fd,fc])
    results += file_path.split("/")[-1] + ",Pd," + str(acc) + "," + str(pd) + "," + str(rd) + "," + str(fd) + "," + str(pc) + "," + str(rc) + "," + str(fc) + ","  + str(hm) + "\n"

    # --- D-C ---
    acc,pd,rd,fd,pc,rc,fc = classify_patients_with_threshold_dev(control_dict, dementia_dict, control_diff_dc_avg, control_diff_dc_std, dementia_diff_dc_avg, dementia_diff_dc_std, 0, file_path.split("/")[-1])
    hm = statistics.harmonic_mean([acc, fd, fc])
    results += file_path.split("/")[-1] + ",D," + str(acc) + "," + str(pd) + "," + str(rd) + "," + str(fd) + "," + str(pc) + "," + str(rc) + "," + str(fc) + "," + str(hm) + (",1" if (control_diff_dc_avg>dementia_diff_dc_avg) else ",0") + (",1" if ((control_diff_dc_avg-0*control_diff_dc_std)>(dementia_diff_dc_avg+0*dementia_diff_dc_std)) else ",0") + "," + str(control_diff_dc_avg) + "," + str(control_diff_dc_std) + "," + str(dementia_diff_dc_avg) + "," + str(dementia_diff_dc_std) +  "\n"
    # --- (D-C)^* ---
    acc,pd,rd,fd,pc,rc,fc = classify_patients_with_threshold_dev(control_dict, dementia_dict, control_diff_dc_avg, control_diff_dc_std, dementia_diff_dc_avg, dementia_diff_dc_std, 2, file_path.split("/")[-1])
    hm = statistics.harmonic_mean([acc, fd, fc])
    results += file_path.split("/")[-1] + ",D*," + str(acc) + "," + str(pd) + "," + str(rd) + "," + str(fd) + "," + str(pc) + "," + str(rc) + "," + str(fc) + "," + str(hm) + (",1" if (control_diff_dc_avg>dementia_diff_dc_avg) else ",0") + (",1" if ((control_diff_dc_avg-2*control_diff_dc_std)>(dementia_diff_dc_avg+2*dementia_diff_dc_std)) else ",0") + "," + str(control_diff_dc_avg) + "," + str(control_diff_dc_std) + "," + str(dementia_diff_dc_avg) + "," + str(dementia_diff_dc_std) +  "\n"

    return results

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
    precisiond = tpd / (tpd + fpd)
    recalld = tpd / (tpd + fnd)
    fd = 2 * (precisiond * recalld) / (precisiond + recalld)
    print("P:", precisiond)
    print("R:", recalld)
    print("F1:", fd)
    print()
    accuracy = ((tpd + tnd) / (tpd + tnd + fpd + fnd))
    print("Accuracy:", (tpd + tnd) / (tpd + tnd + fpd + fnd))
    print()
    print("Detecting Control")
    print("TP:", tpc)
    print("FP:", fpc)
    print("FN:", fnc)
    print("TN:", tnc)
    precisionc = 0 if tpc + fpc == 0 else tpc / (tpc + fpc)
    recallc = 0 if tpc + fnc == 0 else tpc / (tpc + fnc)
    print("P:", precisionc)
    print("R:", recallc)
    fc = 0 if precisionc + recallc == 0 else 2 * (precisionc * recallc) / (precisionc + recallc)
    print("F1:", fc)
    print()
    return accuracy, precisiond,recalld,fd, precisionc, recallc, fc

def classify_patients_with_threshold_dev(control_dict, dementia_dict, tc, devc, td, devd, m, model):
    #
    # tc: threshold (AVG) computed by employing all the transcripts from the Control group
    # devc: threshold (standard deviation) computed by employing all the transcripts from the Control group
    # td: threshold (AVG) computed by employing all the transcripts from the AD group
    # devd: threshold (standard deviation) computed by employing all the transcripts from the AD group
    #

    # Statistics for the AD group
    tpd = 0.0
    fpd = 0.0
    tnd = 0.0
    fnd = 0.0

    # Statistics for the Control group
    tpc = 0.0
    fpc = 0.0
    tnc = 0.0
    fnc = 0.0

    # Merge all the subjects into a single dictionary, by attaching the label to the subject id.
    # The label will be exploited to evaluate the classification.
    subjects = {}
    for s in dementia_dict:
        subjects['d_' + s] = dementia_dict[s]
    for s in control_dict:
        subjects['c_' + s] = control_dict[s]

    tod = "Patient, Y pred, Y true, score, Td, Tc\n"

    for s in subjects:
        # Define subject's thresholds as the global thresholds.
        ttd = td
        ddevd = devd
        ttc = tc
        ddevc = devc

        # Get subject average PPL score.
        subject_avg_ppl = statistics.mean(subjects[s]['d-c'])
        group = s.split("_")[0]
        subject = s.split("_")[1]


        # The following lines of code show that $s$ was held out with the only purpose to rule out her/his contribution
        # from $\overline{D}_{AD}$ or $\overline{D}_{C}$.
        #
        # Control strategy to compute the thresholds:
        # If s belongs to the Control group then we set ttc and ddevc to the scores obtained by excluding s.
        # If s belongs to the AD group then we set ttd and ddevd to the scores obtained by excluding s.
        #
        # PLEASE NOTE:
        # If we are facing an unseen subject we are allowed to adopt global scores td,devd,tc,devc.
        #
        if group == 'd':
            ttd = dementia_dict[subject]['d-c_avg']
            ddevd = dementia_dict[subject]['d-c_std']
        else:
            ttc = control_dict[subject]['d-c_avg']
            ddevc = control_dict[subject]['d-c_std']

        # Define \overline{D}_{C} and \overline{D}_{AD}
        dstarc = ttc - m * ddevc
        dstard = ttd + m * ddevd

        # For each subject s, if:
        #   if |PPL(s) - \overline{D}_{C}| <= |PPL(s) - \overline{D}_{AD}| then we categorize s as Control
        #   if |PPL(s) - \overline{D}_{C}| > |PPL(s) - \overline{D}_{AD}| then we categorize s as Dementia
        if abs(subject_avg_ppl-dstarc) <= abs(subject_avg_ppl-dstard):
            # s has been categorized as Control
            if group == 'c':
                tpc += 1.0
                tnd += 1.0
                tod += s + "," + "C,C," + str(subject_avg_ppl) + "," + str(dstard) + "," + str(dstarc) + "\n"
            else:
                fpc += 1.0
                fnd += 1.0
                tod += s + "," + "C,D," + str(subject_avg_ppl) + "," + str(dstard) + "," + str(dstarc) + "\n"
        else:
            # s has been categorized as Dementia
            if group == 'd':
                tpd += 1.0
                tnc += 1.0
                tod += s + "," + "D,D," + str(subject_avg_ppl) + "," + str(dstard) + "," + str(dstarc) + "\n"
            else:
                fpd += 1.0
                fnc += 1.0
                tod += s + "," + "D,C," + str(subject_avg_ppl) + "," + str(dstard) + "," + str(dstarc) + "\n"

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
    precisiond = 0 if tpd + fpd == 0 else tpd / (tpd + fpd)
    recalld = 0 if tpd + fnd == 0 else tpd / (tpd + fnd)
    fd = 0 if precisiond + recalld == 0 else 2 * (precisiond * recalld) / (precisiond + recalld)
    print("P:", precisiond)
    print("R:", recalld)
    print("F1:", fd)
    print()
    accuracy = ((tpd + tnd) / (tpd + tnd + fpd + fnd))
    print("Accuracy:", (tpd + tnd) / (tpd + tnd + fpd + fnd))
    print()
    print("Detecting Control")
    print("TP:", tpc)
    print("FP:", fpc)
    print("FN:", fnc)
    print("TN:", tnc)
    precisionc = 0 if tpc + fpc == 0 else tpc / (tpc + fpc)
    recallc = 0 if tpc + fnc == 0 else tpc / (tpc + fnc)
    print("P:", precisionc)
    print("R:", recallc)
    fc = 0 if precisionc + recallc == 0 else 2 * (precisionc * recallc) / (precisionc + recallc)
    print("F1:", fc)
    print()

    assert (tpc + fnc) == len(control_dict) and (fpc + tnc) == len(dementia_dict) and (tpd + fnd) == len(dementia_dict) and (fpd + tnd) == len(control_dict)

    open("../resources/data/experiment3/predictions/" + model.replace('.txt','') + "_" + str(m) + "-dev.csv", 'w').write(tod)
    # exit()
    return accuracy, precisiond, recalld, fd, precisionc, recallc, fc


if __name__ == '__main__':
    main()