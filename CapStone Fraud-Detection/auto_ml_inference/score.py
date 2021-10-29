# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd
import joblib

import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType

input_sample = pd.DataFrame({"TransactionDT": pd.Series([0], dtype="int64"), "TransactionAmt": pd.Series([0.0], dtype="float64"), "ProductCD": pd.Series([0], dtype="int64"), "card1": pd.Series([0], dtype="int64"), "card2": pd.Series([0.0], dtype="float64"), "card3": pd.Series([0.0], dtype="float64"), "card4": pd.Series([0], dtype="int64"), "card5": pd.Series([0.0], dtype="float64"), "card6": pd.Series([0], dtype="int64"), "addr1": pd.Series([0.0], dtype="float64"), "addr2": pd.Series([0.0], dtype="float64"), "dist1": pd.Series([0.0], dtype="float64"), "P_emaildomain": pd.Series([0], dtype="int64"), "R_emaildomain": pd.Series([0], dtype="int64"), "C1": pd.Series([0.0], dtype="float64"), "C2": pd.Series([0.0], dtype="float64"), "C4": pd.Series([0.0], dtype="float64"), "C5": pd.Series([0.0], dtype="float64"), "C6": pd.Series([0.0], dtype="float64"), "C7": pd.Series([0.0], dtype="float64"), "C8": pd.Series([0.0], dtype="float64"), "C9": pd.Series([0.0], dtype="float64"), "C10": pd.Series([0.0], dtype="float64"), "C11": pd.Series([0.0], dtype="float64"), "C12": pd.Series([0.0], dtype="float64"), "C13": pd.Series([0.0], dtype="float64"), "C14": pd.Series([0.0], dtype="float64"), "D1": pd.Series([0.0], dtype="float64"), "D2": pd.Series([0.0], dtype="float64"), "D3": pd.Series([0.0], dtype="float64"), "D4": pd.Series([0.0], dtype="float64"), "D5": pd.Series([0.0], dtype="float64"), "D6": pd.Series([0.0], dtype="float64"), "D8": pd.Series([0.0], dtype="float64"), "D9": pd.Series([0.0], dtype="float64"), "D10": pd.Series([0.0], dtype="float64"), "D11": pd.Series([0.0], dtype="float64"), "D12": pd.Series([0.0], dtype="float64"), "D13": pd.Series([0.0], dtype="float64"), "D14": pd.Series([0.0], dtype="float64"), "D15": pd.Series([0.0], dtype="float64"), "M2": pd.Series([0], dtype="int64"), "M3": pd.Series([0], dtype="int64"), "M4": pd.Series([0], dtype="int64"), "M5": pd.Series([0], dtype="int64"), "M6": pd.Series([0], dtype="int64"), "M7": pd.Series([0], dtype="int64"), "M8": pd.Series([0], dtype="int64"), "M9": pd.Series([0], dtype="int64"), "V3": pd.Series([0.0], dtype="float64"), "V4": pd.Series([0.0], dtype="float64"), "V5": pd.Series([0.0], dtype="float64"), "V6": pd.Series([0.0], dtype="float64"), "V7": pd.Series([0.0], dtype="float64"), "V8": pd.Series([0.0], dtype="float64"), "V9": pd.Series([0.0], dtype="float64"), "V10": pd.Series([0.0], dtype="float64"), "V11": pd.Series([0.0], dtype="float64"), "V12": pd.Series([0.0], dtype="float64"), "V13": pd.Series([0.0], dtype="float64"), "V17": pd.Series([0.0], dtype="float64"), "V19": pd.Series([0.0], dtype="float64"), "V20": pd.Series([0.0], dtype="float64"), "V29": pd.Series([0.0], dtype="float64"), "V30": pd.Series([0.0], dtype="float64"), "V33": pd.Series([0.0], dtype="float64"), "V34": pd.Series([0.0], dtype="float64"), "V35": pd.Series([0.0], dtype="float64"), "V36": pd.Series([0.0], dtype="float64"), "V37": pd.Series([0.0], dtype="float64"), "V38": pd.Series([0.0], dtype="float64"), "V40": pd.Series([0.0], dtype="float64"), "V44": pd.Series([0.0], dtype="float64"), "V45": pd.Series([0.0], dtype="float64"), "V46": pd.Series([0.0], dtype="float64"), "V47": pd.Series([0.0], dtype="float64"), "V48": pd.Series([0.0], dtype="float64"), "V49": pd.Series([0.0], dtype="float64"), "V51": pd.Series([0.0], dtype="float64"), "V52": pd.Series([0.0], dtype="float64"), "V53": pd.Series([0.0], dtype="float64"), "V54": pd.Series([0.0], dtype="float64"), "V56": pd.Series([0.0], dtype="float64"), "V58": pd.Series([0.0], dtype="float64"), "V59": pd.Series([0.0], dtype="float64"), "V60": pd.Series([0.0], dtype="float64"), "V61": pd.Series([0.0], dtype="float64"), "V62": pd.Series([0.0], dtype="float64"), "V63": pd.Series([0.0], dtype="float64"), "V64": pd.Series([0.0], dtype="float64"), "V69": pd.Series([0.0], dtype="float64"), "V70": pd.Series([0.0], dtype="float64"), "V71": pd.Series([0.0], dtype="float64"), "V72": pd.Series([0.0], dtype="float64"), "V73": pd.Series([0.0], dtype="float64"), "V74": pd.Series([0.0], dtype="float64"), "V75": pd.Series([0.0], dtype="float64"), "V76": pd.Series([0.0], dtype="float64"), "V78": pd.Series([0.0], dtype="float64"), "V80": pd.Series([0.0], dtype="float64"), "V81": pd.Series([0.0], dtype="float64"), "V82": pd.Series([0.0], dtype="float64"), "V83": pd.Series([0.0], dtype="float64"), "V84": pd.Series([0.0], dtype="float64"), "V85": pd.Series([0.0], dtype="float64"), "V87": pd.Series([0.0], dtype="float64"), "V90": pd.Series([0.0], dtype="float64"), "V91": pd.Series([0.0], dtype="float64"), "V92": pd.Series([0.0], dtype="float64"), "V93": pd.Series([0.0], dtype="float64"), "V94": pd.Series([0.0], dtype="float64"), "V95": pd.Series([0.0], dtype="float64"), "V96": pd.Series([0.0], dtype="float64"), "V97": pd.Series([0.0], dtype="float64"), "V99": pd.Series([0.0], dtype="float64"), "V100": pd.Series([0.0], dtype="float64"), "V126": pd.Series([0.0], dtype="float64"), "V127": pd.Series([0.0], dtype="float64"), "V128": pd.Series([0.0], dtype="float64"), "V130": pd.Series([0.0], dtype="float64"), "V131": pd.Series([0.0], dtype="float64"), "V138": pd.Series([0.0], dtype="float64"), "V139": pd.Series([0.0], dtype="float64"), "V140": pd.Series([0.0], dtype="float64"), "V143": pd.Series([0.0], dtype="float64"), "V145": pd.Series([0.0], dtype="float64"), "V146": pd.Series([0.0], dtype="float64"), "V147": pd.Series([0.0], dtype="float64"), "V149": pd.Series([0.0], dtype="float64"), "V150": pd.Series([0.0], dtype="float64"), "V151": pd.Series([0.0], dtype="float64"), "V152": pd.Series([0.0], dtype="float64"), "V154": pd.Series([0.0], dtype="float64"), "V156": pd.Series([0.0], dtype="float64"), "V158": pd.Series([0.0], dtype="float64"), "V159": pd.Series([0.0], dtype="float64"), "V160": pd.Series([0.0], dtype="float64"), "V161": pd.Series([0.0], dtype="float64"), "V162": pd.Series([0.0], dtype="float64"), "V163": pd.Series([0.0], dtype="float64"), "V164": pd.Series([0.0], dtype="float64"), "V165": pd.Series([0.0], dtype="float64"), "V166": pd.Series([0.0], dtype="float64"), "V167": pd.Series([0.0], dtype="float64"), "V169": pd.Series([0.0], dtype="float64"), "V170": pd.Series([0.0], dtype="float64"), "V171": pd.Series([0.0], dtype="float64"), "V172": pd.Series([0.0], dtype="float64"), "V173": pd.Series([0.0], dtype="float64"), "V175": pd.Series([0.0], dtype="float64"), "V176": pd.Series([0.0], dtype="float64"), "V177": pd.Series([0.0], dtype="float64"), "V178": pd.Series([0.0], dtype="float64"), "V180": pd.Series([0.0], dtype="float64"), "V182": pd.Series([0.0], dtype="float64"), "V184": pd.Series([0.0], dtype="float64"), "V187": pd.Series([0.0], dtype="float64"), "V188": pd.Series([0.0], dtype="float64"), "V189": pd.Series([0.0], dtype="float64"), "V195": pd.Series([0.0], dtype="float64"), "V197": pd.Series([0.0], dtype="float64"), "V200": pd.Series([0.0], dtype="float64"), "V201": pd.Series([0.0], dtype="float64"), "V202": pd.Series([0.0], dtype="float64"), "V203": pd.Series([0.0], dtype="float64"), "V204": pd.Series([0.0], dtype="float64"), "V205": pd.Series([0.0], dtype="float64"), "V206": pd.Series([0.0], dtype="float64"), "V207": pd.Series([0.0], dtype="float64"), "V208": pd.Series([0.0], dtype="float64"), "V209": pd.Series([0.0], dtype="float64"), "V210": pd.Series([0.0], dtype="float64"), "V212": pd.Series([0.0], dtype="float64"), "V213": pd.Series([0.0], dtype="float64"), "V214": pd.Series([0.0], dtype="float64"), "V215": pd.Series([0.0], dtype="float64"), "V216": pd.Series([0.0], dtype="float64"), "V217": pd.Series([0.0], dtype="float64"), "V219": pd.Series([0.0], dtype="float64"), "V220": pd.Series([0.0], dtype="float64"), "V221": pd.Series([0.0], dtype="float64"), "V222": pd.Series([0.0], dtype="float64"), "V223": pd.Series([0.0], dtype="float64"), "V224": pd.Series([0.0], dtype="float64"), "V225": pd.Series([0.0], dtype="float64"), "V226": pd.Series([0.0], dtype="float64"), "V227": pd.Series([0.0], dtype="float64"), "V228": pd.Series([0.0], dtype="float64"), "V229": pd.Series([0.0], dtype="float64"), "V231": pd.Series([0.0], dtype="float64"), "V233": pd.Series([0.0], dtype="float64"), "V234": pd.Series([0.0], dtype="float64"), "V238": pd.Series([0.0], dtype="float64"), "V239": pd.Series([0.0], dtype="float64"), "V242": pd.Series([0.0], dtype="float64"), "V243": pd.Series([0.0], dtype="float64"), "V244": pd.Series([0.0], dtype="float64"), "V245": pd.Series([0.0], dtype="float64"), "V246": pd.Series([0.0], dtype="float64"), "V247": pd.Series([0.0], dtype="float64"), "V249": pd.Series([0.0], dtype="float64"), "V251": pd.Series([0.0], dtype="float64"), "V253": pd.Series([0.0], dtype="float64"), "V256": pd.Series([0.0], dtype="float64"), "V257": pd.Series([0.0], dtype="float64"), "V258": pd.Series([0.0], dtype="float64"), "V259": pd.Series([0.0], dtype="float64"), "V261": pd.Series([0.0], dtype="float64"), "V262": pd.Series([0.0], dtype="float64"), "V263": pd.Series([0.0], dtype="float64"), "V264": pd.Series([0.0], dtype="float64"), "V265": pd.Series([0.0], dtype="float64"), "V266": pd.Series([0.0], dtype="float64"), "V267": pd.Series([0.0], dtype="float64"), "V268": pd.Series([0.0], dtype="float64"), "V270": pd.Series([0.0], dtype="float64"), "V271": pd.Series([0.0], dtype="float64"), "V272": pd.Series([0.0], dtype="float64"), "V273": pd.Series([0.0], dtype="float64"), "V274": pd.Series([0.0], dtype="float64"), "V275": pd.Series([0.0], dtype="float64"), "V276": pd.Series([0.0], dtype="float64"), "V277": pd.Series([0.0], dtype="float64"), "V278": pd.Series([0.0], dtype="float64"), "V279": pd.Series([0.0], dtype="float64"), "V280": pd.Series([0.0], dtype="float64"), "V282": pd.Series([0.0], dtype="float64"), "V283": pd.Series([0.0], dtype="float64"), "V285": pd.Series([0.0], dtype="float64"), "V287": pd.Series([0.0], dtype="float64"), "V288": pd.Series([0.0], dtype="float64"), "V289": pd.Series([0.0], dtype="float64"), "V291": pd.Series([0.0], dtype="float64"), "V292": pd.Series([0.0], dtype="float64"), "V294": pd.Series([0.0], dtype="float64"), "V303": pd.Series([0.0], dtype="float64"), "V304": pd.Series([0.0], dtype="float64"), "V306": pd.Series([0.0], dtype="float64"), "V307": pd.Series([0.0], dtype="float64"), "V308": pd.Series([0.0], dtype="float64"), "V310": pd.Series([0.0], dtype="float64"), "V312": pd.Series([0.0], dtype="float64"), "V313": pd.Series([0.0], dtype="float64"), "V314": pd.Series([0.0], dtype="float64"), "V315": pd.Series([0.0], dtype="float64"), "V317": pd.Series([0.0], dtype="float64"), "V322": pd.Series([0.0], dtype="float64"), "V323": pd.Series([0.0], dtype="float64"), "V324": pd.Series([0.0], dtype="float64"), "V326": pd.Series([0.0], dtype="float64"), "V329": pd.Series([0.0], dtype="float64"), "V331": pd.Series([0.0], dtype="float64"), "V332": pd.Series([0.0], dtype="float64"), "V333": pd.Series([0.0], dtype="float64"), "V335": pd.Series([0.0], dtype="float64"), "V336": pd.Series([0.0], dtype="float64"), "V338": pd.Series([0.0], dtype="float64"), "id_01": pd.Series([0.0], dtype="float64"), "id_02": pd.Series([0.0], dtype="float64"), "id_03": pd.Series([0.0], dtype="float64"), "id_05": pd.Series([0.0], dtype="float64"), "id_06": pd.Series([0.0], dtype="float64"), "id_09": pd.Series([0.0], dtype="float64"), "id_11": pd.Series([0.0], dtype="float64"), "id_12": pd.Series([0], dtype="int64"), "id_13": pd.Series([0.0], dtype="float64"), "id_14": pd.Series([0.0], dtype="float64"), "id_15": pd.Series([0], dtype="int64"), "id_17": pd.Series([0.0], dtype="float64"), "id_19": pd.Series([0.0], dtype="float64"), "id_20": pd.Series([0.0], dtype="float64"), "id_30": pd.Series([0], dtype="int64"), "id_31": pd.Series([0], dtype="int64"), "id_32": pd.Series([0.0], dtype="float64"), "id_33": pd.Series([0], dtype="int64"), "id_36": pd.Series([0], dtype="int64"), "id_37": pd.Series([0], dtype="int64"), "id_38": pd.Series([0], dtype="int64"), "DeviceType": pd.Series([0], dtype="int64"), "DeviceInfo": pd.Series([0], dtype="int64"), "device_name": pd.Series([0], dtype="int64"), "device_version": pd.Series([0], dtype="int64"), "OS_id_30": pd.Series([0], dtype="int64"), "version_id_30": pd.Series([0], dtype="int64"), "browser_id_31": pd.Series([0], dtype="int64"), "version_id_31": pd.Series([0], dtype="int64"), "screen_width": pd.Series([0], dtype="int64"), "screen_height": pd.Series([0], dtype="int64"), "had_id": pd.Series([0.0], dtype="float64"), "TransactionAmt_to_mean_card1": pd.Series([0.0], dtype="float64"), "TransactionAmt_to_std_card1": pd.Series([0.0], dtype="float64"), "TransactionAmt_to_mean_card4": pd.Series([0.0], dtype="float64"), "TransactionAmt_to_std_card4": pd.Series([0.0], dtype="float64"), "TransactionAmt_to_mean_addr1": pd.Series([0.0], dtype="float64"), "TransactionAmt_to_std_addr1": pd.Series([0.0], dtype="float64"), "id_02_to_mean_card1": pd.Series([0.0], dtype="float64"), "id_02_to_std_card1": pd.Series([0.0], dtype="float64"), "id_02_to_mean_card4": pd.Series([0.0], dtype="float64"), "id_02_to_std_card4": pd.Series([0.0], dtype="float64"), "id_02_to_mean_addr1": pd.Series([0.0], dtype="float64"), "id_02_to_std_addr1": pd.Series([0.0], dtype="float64"), "D15_to_mean_card1": pd.Series([0.0], dtype="float64"), "D15_to_std_card1": pd.Series([0.0], dtype="float64"), "D15_to_mean_card4": pd.Series([0.0], dtype="float64"), "D15_to_std_card4": pd.Series([0.0], dtype="float64"), "D15_to_mean_addr1": pd.Series([0.0], dtype="float64"), "D15_to_std_addr1": pd.Series([0.0], dtype="float64"), "TransactionAmt_Log": pd.Series([0.0], dtype="float64"), "TransactionAmt_decimal": pd.Series([0], dtype="int64"), "Transaction_day_of_week": pd.Series([0.0], dtype="float64"), "Transaction_hour": pd.Series([0.0], dtype="float64"), "id_02__id_20": pd.Series([0], dtype="int64"), "id_02__D8": pd.Series([0], dtype="int64"), "D11__DeviceInfo": pd.Series([0], dtype="int64"), "DeviceInfo__P_emaildomain": pd.Series([0], dtype="int64"), "P_emaildomain__C2": pd.Series([0], dtype="int64"), "card2__dist1": pd.Series([0], dtype="int64"), "card1__card5": pd.Series([0], dtype="int64"), "card2__id_20": pd.Series([0], dtype="int64"), "card5__P_emaildomain": pd.Series([0], dtype="int64"), "addr1__card1": pd.Series([0], dtype="int64"), "card1_count_full": pd.Series([0], dtype="int64"), "card2_count_full": pd.Series([0], dtype="int64"), "card3_count_full": pd.Series([0], dtype="int64"), "card4_count_full": pd.Series([0.0], dtype="float64"), "card5_count_full": pd.Series([0], dtype="int64"), "card6_count_full": pd.Series([0.0], dtype="float64"), "id_36_count_full": pd.Series([0.0], dtype="float64"), "id_01_count_dist": pd.Series([0], dtype="int64"), "id_31_count_dist": pd.Series([0.0], dtype="float64"), "id_33_count_dist": pd.Series([0.0], dtype="float64"), "id_36_count_dist": pd.Series([0.0], dtype="float64"), "P_emaildomain_bin": pd.Series([0], dtype="int64"), "P_emaildomain_suffix": pd.Series([0], dtype="int64"), "R_emaildomain_bin": pd.Series([0], dtype="int64"), "R_emaildomain_suffix": pd.Series([0], dtype="int64")})
output_sample = np.array([0])
method_sample = StandardPythonParameterType("predict")

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('method', method_sample, convert_to_provided_type=False)
@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, method="predict"):
    try:
        if method == "predict_proba":
            result = model.predict_proba(data).values
        elif method == "predict":
            result = model.predict(data)
        else:
            raise Exception(f"Invalid predict method argument received ({method})")
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})