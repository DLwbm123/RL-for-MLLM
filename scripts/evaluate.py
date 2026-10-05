import json
import os
from pathlib import Path
import pandas as pd
from src.evaluation import binary_metrics,cluster_bootstrap,fit_calibration
from src.data import write_json

if __name__=='__main__':
    rows=[json.loads(x) for x in Path(os.environ['PREDICTIONS']).read_text().splitlines()]
    frame=pd.DataFrame(rows)
    report=binary_metrics(frame.y,frame.p)
    report['auroc_interval']=cluster_bootstrap(frame,lambda d:binary_metrics(d.y,d.p)['auroc'])
    write_json(os.environ['EVALUATION_OUTPUT'],report)
