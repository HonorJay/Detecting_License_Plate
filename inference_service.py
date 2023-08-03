import logging

import t3qai_client as tc
from t3qai_client import T3QAI_MODULE_PATH, T3QAI_INIT_MODEL_PATH
import os
import shutil
import sys
import pandas as pd
import json
sys.path.append("/work/Detecting_License_Plate")

from Vietnamese.lpdr import main

def init_model():
    """모델 초기화 함수는 _runtime_config.yaml 설정과 맞추어 사용해야 합니다.
    이것은 단순 샘플 소스 입니다.
    """
    ### model load ###
    model = None
    model_path = T3QAI_INIT_MODEL_PATH
    logging.info("model_path : {} ".format(model_path))
    ### inference model load ###
    ### ex) model = learn.load(Path(load_path)) ###
    model_info_dict = {
        "model": model
    }
    return model_info_dict

def inference_dataframe(df, model_info_dict):
    """
    추론함수는 dataframe과 로드한 model을 전달받습니다.
    이것은 단순 샘플 소스 입니다.
    """
    ### ex) model = model_info_dict.get('model')
    ### ex) result = model.predict(df) ###  
    result = []
    logging.info("df data : {} ".format(df))
    logging.info("result : {} ".format(result))
    
    return result

    
def inference_file(files, model_info_dict):
    from t3qai_client import DownloadFile
    result_path = '/work/results.json'

    if len(files) > 1:
        result = {}
        for idx, file in enumerate(files):
            file_name=file.filename
            file_obj=file.file
            new_path = os.path.join('/work/Detecting_License_Plate/', file_name)
            tempFile = open(new_path, 'wb')
            shutil.copyfileobj(file_obj, tempFile)
            tempFile.close()    
            _, results = main(new_path)
            result[idx] = results

        with open(result_path, 'w') as f:
            json.dump(result, f)

        rst = DownloadFile(file_name='results.json', file_path=result_path)
        return rst
            
    else:
        file_name=files[0].filename
        file_obj=files[0].file
        new_path = os.path.join('/work/Detecting_License_Plate/', file_name)
        tempFile = open(new_path, 'wb')
        shutil.copyfileobj(file_obj, tempFile)
        tempFile.close()
        _, results = main(new_path)

        with open(result_path, 'w') as f:
            json.dump(results, f)
        
        rst = DownloadFile(file_name='results.json', file_path=result_path)

        return rst
    
