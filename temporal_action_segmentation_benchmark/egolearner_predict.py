import torch
import torch.nn as nn
import numpy as np


def predict(model, model_dir, results_dir, features_path, vid_list_file,
            feat_suffix, feat_sample_rate, all_sample_rate, epoch,
            actions_dict, device, args):
    # collect arguments
    verbose = args.verbose
    use_best_model = args.use_best_model

    # multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        model.to(device)
        if use_best_model == 'source':
            model.load_state_dict(
                torch.load(model_dir + "/acc_best_source.model"))
            print("load best source model")
        elif use_best_model == 'target':
            model.load_state_dict(
                torch.load(model_dir + "/acc_best_target.model"))
            print("load best target model")
        else:
            model.load_state_dict(
                torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            print("load epoch-" + str(epoch) + " model")

        list_of_videos = []
        if isinstance(vid_list_file, str):
            vid_list_file = [vid_list_file]
        # print("file_list:", vid_list_file)
        # print("feat_suffix:", feat_suffix)
        assert len(feat_suffix) == len(vid_list_file)
        for i, file in enumerate(vid_list_file):
            file_ptr = open(file, 'r')
            list_of_examples = file_ptr.read().strip().split('\n')
            file_ptr.close()
            # print(feat_suffix[i])
            list_of_examples = [
                dict(vid=x,
                     feat_file=
                     f"{features_path}{x.split('.')[0]}{feat_suffix[i]}.pt")
                for x in list_of_examples
            ]
            list_of_videos.extend(list_of_examples)
            # print(file, list_of_examples)

        for vid in list_of_videos:
            if verbose:
                print(vid)
            feat_file = vid["feat_file"]
            # print(feat_file)
            features = torch.load(feat_file)
            features = features.transpose(1, 0)
            features = features[:, ::feat_sample_rate]
            features = features[:, ::all_sample_rate]
            input_x = torch.tensor(features, dtype=torch.float)
            input_x.unsqueeze_(0)
            input_x = input_x.to(device)
            mask = torch.ones_like(input_x)
            predictions, _, _, _, _, _, _, _, _, _, _, _, _, _ = model(
                input_x, input_x, mask, mask, [0, 0], reverse=False)
            _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
            predicted = predicted.squeeze()
            recognition = []
            # print(all_sample_rate,feat_sample_rate)
            for i in range(predicted.size(0)):
                recognition = np.concatenate((recognition, [
                    list(actions_dict.keys())[list(
                        actions_dict.values()).index(predicted[i].item())]
                ] * all_sample_rate * feat_sample_rate))
            f_name = vid["vid"].split('/')[-1].split('.')[0]
            f_ptr = open(results_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
