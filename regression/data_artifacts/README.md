# Data artifacts

`real_proxy_base/` contains the frozen split and long-range proxy caches used by
the final real-data run. It does not contain source videos or heat-chamber data.
The two caches were generated independently for `turbulence_sequences` and
`turbulence_sequences_RLRAT` and are consumed by `real/train_real_long_logj.py`.

Raw datasets are not redistributed. Place them under `data/` as described in the
top-level README.
