import sys

def report_info(records, keys=None):
    if keys == None:
        keys = ['epoch', 'avg_loss', 'max_loss', 'train_acc', 'test_acc', 'mAP',
                'best_test_acc', 'best_test_epoch', 'best_mAP', 'best_mAP_epoch']
    tmp_list = []
    for key in keys:
        if key in records.keys():
            value = records[key]
            tmp_str = key + \
                ':{:.4f}. '.format(value) if isinstance(
                    value, float) else key + ':{}. '.format(value)
            tmp_list.append(tmp_str)
    final_str = '   '.join(tmp_list)
    print(final_str)


def record_info(record_file_name, records, keys):
    strs = []
    for key in keys:
        strs.append(key + ": " + str(records[key]))
    with open(record_file_name, 'a') as f:
        f.write("{" + ','.join(strs) + "}\n")
        f.close()


def get_mAP(probs, labels):
    assert len(probs) == len(labels)
    print("getting mAP.")
    ap = []
    for prob, label in zip(probs, labels):
        num_gt = len(label[label==1])
        prob, label = sort_prob_label(prob, label)
        prev_recall = 0.0
        prev_precision = 1.0
        counter = 0.0
        tmp_ap = 0.0
        for i, l in enumerate(label):
            if l == 1:
                counter += 1.0
            precision = counter / (i+1)
            recall = counter / num_gt
            tmp_ap += (recall - prev_recall) * ((precision + prev_precision) / 2)
            prev_precision = precision
            prev_recall = recall
            if counter == num_gt:
                break
        ap.append(tmp_ap)
    assert len(probs) == len(ap)
    return sum(ap) / len(ap)


def sort_prob_label(prob, label):
    tmp = zip(prob, label)
    sorted_tmp = sorted(tmp, key=lambda x:x[0], reverse=True)
    tuples = zip(*sorted_tmp)
    prob, label = (list(t) for t in tuples)
    return prob, label


def split_dataset(source_data, cnfs, percent=None, few_shot_num=None):
    if percent == None and few_shot_num == None:
        print("Error! one of percent and few shot num should be set.")
        sys.exit(-1)
    if cnfs["dataset"] == "EMS" or cnfs["dataset"] == "ENCAA":
        lbl_data, ulbl_data, test, val = source_data["lbl_data"], source_data[
            "ulbl_data"], source_data["test"], source_data["val"]
        if cnfs["dataset"] == "EMS":
            event_num = 7
        if cnfs["dataset"] == "ENCAA":
            event_num = 1
        event_data = [[] for i in range(event_num)]
        for tmp_data in lbl_data:
            event_data[tmp_data["event_label"]].append(tmp_data)
        new_lbl_data = []
        for i in range(event_num):
            if percent is not None:
                add_num = int(len(event_data[i]) * percent)
            else:
                add_num = few_shot_num
            new_lbl_data.extend(event_data[i][:add_num])
            ulbl_data.extend(event_data[i][add_num:])
        new_source_data = {}
        new_source_data["lbl_data"], new_source_data["ulbl_data"], new_source_data[
            "test"], new_source_data["val"] = new_lbl_data, ulbl_data, test, val
        return new_source_data
    if cnfs["dataset"] == "MS" or cnfs["dataset"] == "NCAA":
        train, test, val = source_data["train"], source_data["test"], source_data["val"]
        if cnfs["dataset"] == "MS":
            event_num = 7
        elif cnfs["dataset"] == "NCAA":
            event_num = 10
        event_data = [[] for i in range(event_num)]
        for tmp_data in train:
            event_data[tmp_data["event_label"]].append(tmp_data)
        new_train = []
        for i in range(event_num):
            if percent is not None:
                add_num = int(len(event_data[i]) * percent)
            else:
                add_num = few_shot_num
            new_train.extend(event_data[i][:add_num])
            val.extend(event_data[i][add_num:])
        new_source_data = {}
        new_source_data["train"], new_source_data["test"], new_source_data["val"] = new_train, test, val
        return new_source_data
        

def resume_records(record_file, epoch_num=0, iter_num=None):
    if iter_num is not None:
        pattern = "iter: {},epoch: {}".format(iter_num, epoch_num)
    else:
        pattern = "epoch: {}".format(epoch_num)
    new_lines = []
    is_find = False
    with open(record_file, 'r') as f:
        for line in f:
            new_lines.append(line)
            if line.find(pattern) != -1:
                is_find=True
                break
    if not is_find:
        print("do not find this line of record, try again.")
        sys.exit(-1)
    with open(record_file, 'w+') as f:
        f.writelines(new_lines)

