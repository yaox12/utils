from subprocess import call
from pathlib import Path
from tqdm import tqdm

base_dir = '/home/admin/workspace/shared/GLD_v2_tfrecord/'
idx_path = Path(base_dir, 'train_idx_files')
tfrecord2idx_script = "tfrecord2idx"

if not idx_path.exists():
    idx_path.mkdir()

for i in tqdm(range(1000)):
    tfrecord = Path(base_dir, 'train', f'tf_record.{i}')
    tfrecord_idx = Path(idx_path, f'tf_record.{i}.idx')
    if not tfrecord_idx.is_file():
        call([tfrecord2idx_script, tfrecord, tfrecord_idx])
