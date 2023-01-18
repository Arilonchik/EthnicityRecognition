import os
from FER.FER_model import FERModel
from multiprocessing import Pool
import shutil
import json


def filter_async(input_dir, output_dir, model_config, device, return_meta):
    file_list = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
    recogniser = FERModel(model_config, device)
    results = []

    for file in file_list:
        results.append(recogniser.process_image(file))

    # with Pool(8) as p:
    #     results = p.map(recogniser.process_image, file_list)

    def proc_file(label, filepath):
        if len(label) > 1:
            label = "_".join(label)
        else:
            label = label[0]
        done_dir = os.path.join(output_dir, label)
        os.makedirs(done_dir, exist_ok=True)
        shutil.move(filepath, done_dir)

    print(results)
    if not return_meta:
        with Pool(8) as p:
            p.map(proc_file, results)
    else:
        meta = {image: labels for labels, image in results}
        json.dump(meta, open(os.path.join(output_dir, "meta.json"), "w"), indent=4)






