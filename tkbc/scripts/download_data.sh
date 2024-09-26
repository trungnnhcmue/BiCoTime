# Copyright (c) Facebook, Inc. and its affiliates.

# Download ICEWS14, ICEWS05-15, yago15k and wikidata

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd "$DIR"

wget https://dl.fbaipublicfiles.com/tkbc/data.tar.gz
tar -xvzf data.tar.gz
tar -xvzf wiki12k_yago11k.tar.gz -C src_data/
rm data.tar.gz
