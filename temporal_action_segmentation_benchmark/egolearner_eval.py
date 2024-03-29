import numpy as np
import argparse
import os
from egobridge_settings import get_annotations_from_settings

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float64)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(
            p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_data', default='data/')
    parser.add_argument('--path_result', default='results/')
    parser.add_argument('--dataset', default="egobridge")
    parser.add_argument("--exp_type",
                        type=str,
                        default="ego-only",
                        choices=[
                            "ego-only", "exo-only", "ego2exo", "exo2ego",
                            "ego-only-gazed", "ego-only-center",
                            'ego-exo-cotraining-ego', 'ego-exo-cotraining-exo', 
                            'ego-exo-cotraining-gazed-ego', 'ego-exo-cotraining-gazed-exo', 
                            'ego-exo-da-exo','exo-ego-da-ego',
                            'ego-exo-gazed-da-exo','exo-ego-gazed-da-ego',
                        ])
    parser.add_argument(
        "--test",
        action="store_true",
    )

    args = parser.parse_args()
    assert args.dataset == "egobridge"

    split = "test" if args.test else "val"
    ground_truth_path = os.path.join(args.path_data, "gts_fps25/")
    recog_path = os.path.join(args.path_result, split)

    train_source_vid_list_file,train_source_feat_suffix, test_source_vid_list_file,test_source_val_feat_suffix, \
            train_target_vid_list_file, train_target_feat_suffix, test_target_vid_list_file,test_target_feat_suffix = get_annotations_from_settings(args)

    vid_list_file = test_target_vid_list_file

    list_of_videos = []
    if isinstance(vid_list_file, str):
        vid_list_file = [vid_list_file]
    # print("file_list:", vid_list_file)
    for i, file in enumerate(vid_list_file):
        file_ptr = open(file, 'r')
        list_of_examples = file_ptr.read().strip().split('\n')
        file_ptr.close()
        list_of_examples = [x for x in list_of_examples]
        list_of_videos.extend(list_of_examples)

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        recog_file = os.path.join(recog_path, vid.split('.')[0])
        recog_content = read_file(recog_file).split('\n')[1].split()
        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    # print(" ")
    sum = 0
    # print("Acc: %.4f" % (100 * float(correct) / total))
    # print('Edit: %.4f' % ((1.0 * edit) / len(list_of_videos)))
    sum += (100 * float(correct) / total)
    sum += ((1.0 * edit) / len(list_of_videos))
    sum_f1 = 0
    for s in range(len(overlap)):
        precision = 0.0 if (tp[s] +
                            fp[s]) == 0 else tp[s] / float(tp[s] + fp[s])
        recall = 0.0 if (tp[s] + fn[s]) == 0 else tp[s] / float(tp[s] + fn[s])

        f1 = 0.0 if (precision +
                     recall) == 0.0 else 2.0 * (precision *
                                                recall) / (precision + recall)
        f1 = np.nan_to_num(f1) * 100
        # print('F1@%0.2f: %.4f' % (overlap[s], f1))
        sum += f1
        sum_f1 += f1
    print(
        f"Acc: {100 * float(correct) / total:.4f}, Edit: {(1.0 * edit) / len(list_of_videos)}, F1@Avg: {sum_f1 / 3:.4f}, Avg: {sum / 5:.4f}"
    )
    # print("F1@Avg: %.4f" % (sum_f1 / 3))
    # print("Avg: %.4f" % (sum / 5))


if __name__ == '__main__':
    
    main()
