# utils

some useful utils

## dataset

- [`tfrecord_index.py`](tfrecord_index.py)
  - Create index files for tfrecords
  - The used script `tfrecord2idx` is shipped with Nvidia [DALI](https://github.com/NVIDIA/DALI)
- [`string_to_int.py`](string_to_int.py)
  - Convert `label` in tfrecords from `string(byte)` to `int` with multiprocesssing
  - When to use: Some scripts convert raw images to tfrecords but store `label` of examples as `string (byte)` type, which cannot be parsed correctly by DALI. This script may be helpful.
