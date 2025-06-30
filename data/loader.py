import wfdb
import os

def scarica_record(record_id, base_dir="data/mitbih"):
    os.makedirs(base_dir, exist_ok=True)
    rec_path = os.path.join(base_dir, record_id)
    if not os.path.exists(rec_path + ".dat"):
        wfdb.dl_database("mitdb", dl_dir=base_dir, records=[record_id])
