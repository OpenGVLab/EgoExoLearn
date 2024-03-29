EgoExoLearn Annotations 
This repository contains the annotations for the EgoExoLearn dataset.

### :mortar_board: Note
This is the complete annotation. The annotations of each benchmark is derived from the complete annotations, and are separatedly placed in the benchmark folders for the easy of use.

We release the annotation for the training and validation set only. We will separately 

## :fire: Updates <a name="news"></a>
Annotations of the fine-level actions are released.

## Contents
`fine_annotation_trainval_en.csv`: The annotation csv file. It include the following fields:
- `video_uid`: The uid of the video. This is the same with the downloaded videos.
- `annotation_uid`: An unique ID for each annotation.
- `subset`: Indicates whether this annotation belongs to the `train` or `val` subset.
- `view`: Indicates whether this video is in egocentric view or exocentric view.
- `scene`: We broadly split the recording scenes into `kitchen` and `lab`.
- `start_sec`: The start timestamp of this annotation.
- `end_sec`: The end timestamp of this annotation.
- `narration_en`: The manual description annotation for the video between `start_sec` and `end_sec`.
- `narration_en_hand_prompt`: We seperate the description of the left and right hands by prompting GPT 3.5. These captions can be used for furtuer researches related with detailed hand analysis.
- `narration_en_no_hand_prompt`: We use GPT 3.5 to remove the left hand and right hand in the `narration_en` descriptions. These captions become more natural.

`narration_noun_taxonomy.csv` and `narration_verb_taxonomy.csv`: We use these taxonomy to extract verb and noun IDs in the action anticipation and planning benchmarks.

:mailbox_with_mail: For any questions, please contact: [Yifei Huang]((https://hyf015.github.io/)) ( hyf at iis.u-tokyo.ac.jp ) 