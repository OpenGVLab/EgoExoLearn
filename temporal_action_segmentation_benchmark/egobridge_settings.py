import os.path as osp

def get_annotations_from_settings(args):
    splits_path = osp.join(args.path_data, "splits")
    print("use experiment type:", args.exp_type)
    if args.exp_type == "ego-only":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle")]
        train_source_feat_suffix = [""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"ego_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle")]
        train_target_feat_suffix = [""]

        test_target_vid_list_file = [osp.join(splits_path,"ego_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"ego_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
            
    elif args.exp_type == "ego-only-gazed":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle")]
        train_source_feat_suffix = [""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"ego_gaze_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle")]
        train_target_feat_suffix = [""]

        test_target_vid_list_file = [osp.join(splits_path,"ego_gaze_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"ego_gaze_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "exo-only":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"exo_train_annotations.bundle")]
        train_source_feat_suffix = [""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"exo_train_annotations.bundle")]
        train_target_feat_suffix = [""]

        test_target_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"exo_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "ego-exo-cotraining-ego":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_source_feat_suffix = ["",""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"ego_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_target_feat_suffix = ["",""]

        test_target_vid_list_file = [osp.join(splits_path,"ego_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"ego_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "ego-exo-cotraining-exo":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_source_feat_suffix = ["",""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_target_feat_suffix = ["",""]

        test_target_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"exo_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "ego-exo-cotraining-gazed-ego":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_source_feat_suffix = ["",""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"ego_gaze_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_target_feat_suffix = ["",""]

        test_target_vid_list_file = [osp.join(splits_path,"ego_gaze_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"ego_gaze_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "ego-exo-cotraining-gazed-exo":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_source_feat_suffix = ["",""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle"),
                                      osp.join(splits_path,"exo_train_annotations.bundle")]
        train_target_feat_suffix = ["",""]

        test_target_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"exo_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "ego-exo-da-exo":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle")]
        train_source_feat_suffix = [""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"ego_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"exo_train_annotations.bundle")]
        train_target_feat_suffix = [""]

        test_target_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"exo_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "exo-ego-da-ego":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"exo_train_annotations.bundle")]
        train_source_feat_suffix = [""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_train_annotations.bundle")]
        train_target_feat_suffix = [""]

        test_target_vid_list_file = [osp.join(splits_path,"ego_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"ego_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "ego-exo-gazed-da-exo":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle")]
        train_source_feat_suffix = [""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"ego_gaze_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"exo_train_annotations.bundle")]
        train_target_feat_suffix = [""]

        test_target_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"exo_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    elif args.exp_type == "exo-ego-gazed-da-ego":
        # train source -> ego-train
        train_source_vid_list_file = [osp.join(splits_path,"exo_train_annotations.bundle")]
        train_source_feat_suffix = [""]

        # val source -> ego-val
        test_source_vid_list_file = [osp.join(splits_path,"exo_val_annotations.bundle")]
        test_source_val_feat_suffix = [""]

        # train target -> ego-train
        train_target_vid_list_file = [osp.join(splits_path,"ego_gaze_train_annotations.bundle")]
        train_target_feat_suffix = [""]

        test_target_vid_list_file = [osp.join(splits_path,"ego_gaze_val_annotations.bundle")]
        test_target_feat_suffix = [""]
        if args.test:
            print("runing test mode")
            test_source_vid_list_file = [osp.join(splits_path,"ego_gaze_test_annotations.bundle")]
            test_source_val_feat_suffix = [""]
    else:
        raise NotImplementedError

    return train_source_vid_list_file, train_source_feat_suffix, test_source_vid_list_file, test_source_val_feat_suffix, \
            train_target_vid_list_file, train_target_feat_suffix, test_target_vid_list_file, test_target_feat_suffix
    
    