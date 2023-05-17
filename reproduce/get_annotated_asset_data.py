#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Reads in the human ratings provided in ASSET and outputs a JSONL file in the format expected by UniEval.

[
    {
        "doc_id": "dm-test-8764fb95bfad8ee849274873a92fb8d6b400eee2",
        "system_id": "M11",
        "source": "Paul Merson has restarted his row with Andros Townsend after the Tottenham midfielder was brought on with only seven minutes remaining in his team 's 0-0 draw with Burnley on Sunday . 'Just been watching the game , did you miss the coach ? # RubberDub # 7minutes , ' Merson put on Twitter . Merson initially angered Townsend for writing in his Sky Sports column that 'if Andros Townsend can get in ( the England team ) then it opens it up to anybody . ' Paul Merson had another dig at Andros Townsend after his appearance for Tottenham against Burnley Townsend was brought on in the 83rd minute for Tottenham as they drew 0-0 against Burnley Andros Townsend scores England 's equaliser in their 1-1 friendly draw with Italy in Turin on Tuesday night The former Arsenal man was proven wrong when Townsend hit a stunning equaliser for England against Italy and he duly admitted his mistake . 'It 's not as though I was watching hoping he would n't score for England , I 'm genuinely pleased for him and fair play to him – it was a great goal , ' Merson said . 'It 's just a matter of opinion , and my opinion was that he got pulled off after half an hour at Manchester United in front of Roy Hodgson , so he should n't have been in the squad . 'When I 'm wrong , I hold my hands up . I do n't have a problem with doing that - I 'll always be the first to admit when I 'm wrong . ' Townsend hit back at Merson on Twitter after scoring for England against Italy Sky Sports pundit Merson ( centre ) criticised Townsend 's call-up to the England squad last week Townsend hit back at Merson after netting for England in Turin on Wednesday , saying 'Not bad for a player that should be 'nowhere near the squad ' ay @ PaulMerse ? ' Any bad feeling between the pair seemed to have passed but Merson was unable to resist having another dig at Townsend after Tottenham drew at Turf Moor .",
        "reference": "Andros Townsend an 83rd minute sub in Tottenham 's draw with Burnley . He was unable to find a winner as the game ended without a goal . Townsend had clashed with Paul Merson last week over England call-up .",
        "system_output": "Paul merson was brought on with only seven minutes remaining in his team 's 0-0 draw with burnley . Andros townsend scored the tottenham midfielder in the 89th minute . Paul merson had another dig at andros townsend after his appearance . The midfielder had been brought on to the england squad last week . Click here for all the latest arsenal news news .",
        "scores": {
            "coherence": 1.3333333333333333,
            "consistency": 1.0,
            "fluency": 3.0,
            "relevance": 1.6666666666666667,
            "overall": 1.75
        }
    },
    ...
]


Example call (from UniEval/reproduce/):
    python get_annotated_asset_data.py

"""

import json
import pandas as pd
import random
from pathlib import Path

random.seed(42)

asset_dir = '/srv/scratch1/kew/ats/data/en/asset' # adjust this path as needed
outfile = 'data/simplification/asset.json'

Path(outfile).parent.mkdir(parents=True, exist_ok=True)

# collect references from asset (not included in the human ratings csv)
refs_sents = []
for i in range(0, 10):
    asset_reference_file = Path(asset_dir) / f'dataset' / f'asset.test.simp.{i}'
    with open(asset_reference_file, 'r', encoding='utf8') as f:
        refs_sents.append([line.strip() for line in f.readlines()])
refs_sents = list(map(list, [*zip(*refs_sents)])) # transpose from [# samples, # refs_per_sample] to [# refs_per_sample, # samples]
# refs_sents = refs_sents[0]

# load human ratings csv
hr_file = Path(asset_dir) / f'human_ratings' / f'human_ratings.csv'
df = pd.read_csv(hr_file)

annotated_data = []
for i, sdf in df.groupby(['worker_id', 'original_sentence_id']):
    data = {
        "source": sdf['original'].iloc[0],
        "system_output": sdf['simplification'].iloc[0],
        "reference": "",
        "system_id": "",
        "doc_id": int(sdf['original_sentence_id'].iloc[0]),
        "scores": {
            "coherence": float(sdf[sdf.aspect == 'meaning']['rating'].item()),
            "consistency": float(sdf[sdf.aspect == 'meaning']['rating'].item()),
            "fluency": float(sdf[sdf.aspect == 'fluency']['rating'].item()),
            "simplicity": float(sdf[sdf.aspect == 'simplicity']['rating'].item()),
            # "relevance": 0.0, # relevance asks "Is this summary relevant to the reference?", but human ratings do not consider references, therefore it's irrelevant here!
            # "overall": 0.0,
        }
    }

    # get references using doc_id
    data['reference'] = random.choice(refs_sents[data['doc_id']])
    annotated_data.append(data)


# sort data by doc_id
annotated_data = sorted(annotated_data, key=lambda x: x['doc_id'])

with open(outfile, 'w', encoding='utf8') as f:
    json.dump(annotated_data, f, indent=4, ensure_ascii=False)

print(f'Collected {len(annotated_data)} annotated data points. See {outfile}')