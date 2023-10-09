from datetime import datetime
import yaml
from torch.nn import DataParallel
from packages.utils import init, ckpt, train_eval_test, tools
from packages.utils.data.dataset import VIPDatasetNewTransforms
from PIL import ImageFile
import resource
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True


def trn_sgl_epoch(model, p_model, cnfs, trnldr, tstldr, valldr, record_file, optimizer, records):
    records['train_acc'], records['avg_loss'], records['max_loss'] = train_eval_test.train(
        trnldr, p_model, cnfs, optimizer)
    need_eval = records["epoch"] % cnfs["log_inter"] == 0
    is_last_epoch = records["epoch"] == cnfs["epoch"]
    if need_eval or is_last_epoch:
        records['mAP'] = train_eval_test.eval_or_test(tstldr, model, cnfs)
        records['val_mAP'] = train_eval_test.eval_or_test(valldr, model, cnfs)
        
        report_keys = ['epoch', 'avg_loss', 'max_loss', 'train_acc', 'mAP', "val_mAP"]
        records_keys = ['epoch', 'avg_loss', 'max_loss', 'train_acc', 'mAP', "val_mAP"]
        tools.report_info(records, report_keys)
        tools.record_info(record_file, records, records_keys)


if __name__ == "__main__":
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (65535, rlimit[1]))
    cmd_args = init.init_parser()
    cnfs = yaml.load(open(cmd_args.config, 'r'), Loader=yaml.Loader)
    init.init_seed(cnfs['seed'])
    cnfs["gpu_id"] = cmd_args.gpu_id
    init.init_paths_and_cuda(cnfs)
    source_data = init.init_data(cnfs)
    source_data = tools.split_dataset(
        source_data, cnfs, cnfs["percent"])
    train, val, test = source_data["train"], source_data["val"], source_data["test"]

    model = init.init_model(cnfs["model"]["type"], cnfs["models_path"], cnfs)
    model = model.cuda()
    p_model = DataParallel(model).cuda()

    optimizer, scheduler = init.init_optimizer_and_scheduler(cnfs, model)
    records = init.init_record_dict()

    if cmd_args.eval:
        cnfs['resume_model'] = cmd_args.resume_model
        start_epoch, scheduler, model, optimizer, records = ckpt.resume_model(
            cnfs, model, scheduler, optimizer, records, False)
        trainset = VIPDatasetNewTransforms(train, cnfs, True)
        testset = VIPDatasetNewTransforms(test, cnfs)
        valset = VIPDatasetNewTransforms(val, cnfs)
        trnldr = init.init_loader(cnfs, trainset, cnfs["batch_size"], is_train=True)
        tstldr = init.init_loader(cnfs, testset, 1)
        valldr = init.init_loader(cnfs, valset, 1)
        mAP = train_eval_test.eval_or_test(tstldr, model, cnfs)
        val_mAP = train_eval_test.eval_or_test(valldr, model, cnfs)
        print("test_mAP: {:.4f}, val_mAP: {:.4f}".format(mAP, val_mAP))
        
        sys.exit(0)

    record_file = init.init_record(cnfs, cmd_args.is_resume, cmd_args.force)

    if cmd_args.is_resume:
        cnfs["resume_model"] = cmd_args.resume_model
        start_epoch, scheduler, model, optimizer, records = ckpt.resume_model(
            cnfs, model, scheduler, optimizer, records, False)
        if cmd_args.resume_records:
            tools.resume_records(record_file, start_epoch-1)
        trainset = VIPDatasetNewTransforms(train, cnfs, True)
        testset = VIPDatasetNewTransforms(test, cnfs)
        valset = VIPDatasetNewTransforms(val, cnfs)
        trnldr = init.init_loader(cnfs, trainset, cnfs["batch_size"], is_train=True)
        tstldr = init.init_loader(cnfs, testset, 1)
        valldr = init.init_loader(cnfs, valset, 1)
    else:
        start_epoch = 1
        trainset = VIPDatasetNewTransforms(train, cnfs, True)
        testset = VIPDatasetNewTransforms(test, cnfs)
        valset = VIPDatasetNewTransforms(val, cnfs)
        trnldr = init.init_loader(cnfs, trainset, cnfs["batch_size"], is_train=True)
        tstldr = init.init_loader(cnfs, testset, 1)
        valldr = init.init_loader(cnfs, valset, 1)

    del_files = []
    for i in range(start_epoch, cnfs["epoch"] + 1):
        records['epoch'] = i

        trn_sgl_epoch(model, p_model, cnfs, trnldr, tstldr, valldr,
                      record_file, optimizer, records)
        need_save = i % cnfs["save_inter"] == 0
        is_last_epoch = i == cnfs["epoch"]
        if need_save or is_last_epoch:
            tmp_records = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()}
            tmp_records.update(records)
            if not is_last_epoch:
                if len(del_files) >= cnfs["max_ckpt_num"]:
                    tmp_file = ckpt.save_checkpoint(
                        tmp_records, cnfs["model"]["name"], cnfs)
                    ckpt.del_tmp_ckpts(del_files)
                    del_files = [tmp_file]
                else:
                    del_files.append(ckpt.save_checkpoint(
                        tmp_records, cnfs["model"]["name"], cnfs))
            else:
                ckpt.save_checkpoint(
                    tmp_records, cnfs["model"]["name"], cnfs)
                ckpt.del_tmp_ckpts(del_files)
        scheduler.step()
    tmp_records = {"end_time": datetime.today()}
    tools.record_info(record_file, tmp_records, tmp_records.keys())
