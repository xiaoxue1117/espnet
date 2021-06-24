#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000
data_type=legacy

log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

if [ -z "${TEDLIUM3}" ]; then
    log "Fill the value of 'TEDLIUM3' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    if [ ! -e "${TEDLIUM3}" ]; then
	    echo "stage 1: Data Download to ${TEDLIUM3}"
        echo "downloading TEDLIUM_release3 data (it won't re-download if it was already downloaded.)"
        # the following command won't re-get it if it's already there
        # because of the --continue switch.
        wget --continue http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz || exit 1
        tar xf "TEDLIUM_release-3.tgz"
    else
        echo "$0: not downloading or un-tarring TEDLIUM_release2 because it already exists."
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    num_sph=$(find -L TEDLIUM_release-3/legacy -name '*.sph' | wc -l)
    # We mainly use TED-LIUM 3 "legacy" distribution, on which the dev and test datasets are the same as in TED-LIUM 2 (and TED-LIUM1).
    # It contains 2351 sph files for training and 19 sph files for dev/test (total 2370).
    # Because the "legacy" contains symbolic links to "data", we use `find -L`.
    if [ "$num_sph" != 2370 ]; then
        echo "$0: expected to find 2370 .sph files in the directory db/TEDLIUM_release3/legacy, found $num_sph"
        exit 1
    fi
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    log "stage 3: Data Preparation"
    local/prepare_data.sh ${data_type}
    for dset in dev test train; do
        utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
    done
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    log "stage 4: LM Data Preparation"
    lmdatadir=data/local/other_text
    mkdir -p ${lmdatadir}
    gunzip -c db/TEDLIUM_release-3/LM/*.en.gz | sed 's/ <\/s>//g' | local/join_suffix.py | \
        awk '{printf("tedlium3_lng_%08d %s\n", NR, $0) }' > ${lmdatadir}/text
fi
