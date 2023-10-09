import os
import argparse
import pickle
import json
import scipy.io as scio

import numpy as np
import PIL.Image as Image
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data(img_path, data, data_name, need_multi_vip):
    if data_name == "NCAA":
        event_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9}
    elif data_name == "MS":
        event_map = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6}
    ret = []
    for tmp_data in tqdm(data, ncols=80):
        img_id = tmp_data["name"][0][-8:-4]
        tmp_dict = {}
        tmp_dict["img_id"] = img_id
        tmp_dict["img_path"] = os.path.join(img_path, img_id + '.jpg')
        img = Image.open(tmp_dict["img_path"]).convert("RGB")
        width, height = img.size
        if data_name == "MS":
            faces = tmp_data["Face"][0]
        elif data_name == "NCAA":
            faces = tmp_data["player"][0]
        tmp_dict["event_label"] = event_map[faces[0]["eventLabel"][0][0]]
        bboxes = []
        labels = []
        for face in faces:
            bbox = face["rect"]
            bbox.resize(4,)
            bbox = bbox.tolist()
            label = face["label"][0][0]
            if int(label) !=0 and int(label) != 1:
                label = 1
            labels.append(int(label))
            bboxes.append(bbox)
        bboxes, labels = calib_bboxes(bboxes, width, height, data_name, labels)
        tmp_lbls = np.array(labels, dtype=np.int64)
        if len(tmp_lbls) == len(tmp_lbls[tmp_lbls==0]):
            print("Img:{} has no imp person.".format(img_id))
            continue
        if not need_multi_vip:
            if len(tmp_lbls[tmp_lbls==1]) != 1:
                continue
        tmp_dict["labels"] = tmp_lbls.tolist()
        tmp_dict["bboxes"] = bboxes
        ret.append(tmp_dict)
    return ret


def calib_bboxes(bboxes, width, height, data_name, labels):
    ret = []
    ret_labels = []
    for label, bbox in zip(labels, bboxes):
        lt_bbox = bbox_2_left_top(bbox, data_name)
        x, y, w, h = lt_bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        if w <= 0 or h<= 0:
            print("empty bbox")
            continue
        if x< 0 or y<0 or x + w > width or y + h > height:
            print("bbox out of bound")
            continue
        ret_labels.append(label)
        ret.append([x, y, w, h])
    return ret, ret_labels

def bbox_2_left_top(bbox, data_name):
    if data_name == "MS":
        x, y, w, h = bbox
        x = x - w // 2
        y = y - h // 2
    elif data_name == "NCAA":
        return bbox
    return [x, y, w, h]


def preprocess_dataset(data_name, mat_path, img_path, need_multi_vip):
    data = scio.loadmat(mat_path)
    train, val, test = data["train"][0], data["val"][0], data["test"][0]
    num_train, num_val, num_test = len(train), len(val), len(test)
    print("train:{}, val:{}, test:{}".format(num_train, num_val, num_test))
    train = get_data(os.path.join(img_path, 'train'), train, data_name, need_multi_vip)
    val = get_data(os.path.join(img_path, 'val'), val, data_name, need_multi_vip)
    test = get_data(os.path.join(img_path, 'test'), test, data_name, need_multi_vip)
    print("train:{}, val:{}, test:{}".format(len(train), len(val), len(test)))
    return {"train": train, "val": val, "test": test}


def set_paths(dataset):
    if dataset == "MS":
        mat_path = "/data/yukun/MS/data/annotations"
        img_path = "/data/yukun/MS/images"
        save_path = "/data/yukun/MS/data"
    elif dataset == "NCAA":
        mat_path = "/data/yukun/NCAA/data/annotations"
        img_path = "/data/yukun/NCAA/Images"
        save_path = "/data/yukun/NCAA/data"
    return mat_path, img_path, save_path
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MS")
    parser.add_argument('--need_multi_vip', action="store_true")
    args = parser.parse_args()

    mat_path, img_path, save_path = set_paths(args.dataset)
    save_dict = preprocess_dataset(args.dataset, mat_path, img_path, args.need_multi_vip)
    if args.need_multi_vip:
        pickle_file_path = os.path.join(save_path, "processed(multi).pkl")
        json_file_path = os.path.join(save_path, "processed(multi).json")
    else:
        pickle_file_path = os.path.join(save_path, "processed.pkl")
        json_file_path = os.path.join(save_path, "processed.json")
    pickle.dump(save_dict, open(pickle_file_path, "wb+"))
    json.dump(save_dict, open(json_file_path, "w+", encoding="utf-8"), indent=4)
